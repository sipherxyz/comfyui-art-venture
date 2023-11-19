import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

const supportedNodes = ["LoadImageFromUrl", "LoadImageAsMaskFromUrl", "LoadVideoFromUrl"]

function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}

function injectHidden(widget) {
  widget.computeSize = (target_width) => {
    if (widget.hidden) {
      return [0, -4];
    }
    return [target_width, 20];
  };
  widget._type = widget.type
  Object.defineProperty(widget, "type", {
    set: function (value) {
      widget._type = value;
    },
    get: function () {
      if (widget.hidden) {
        return "hidden";
      }
      return widget._type;
    }
  });
}

function useKVState(nodeType) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    chainCallback(this, "onConfigure", function (info) {
      if (!this.widgets) {
        //Node has no widgets, there is nothing to restore
        return
      }
      if (typeof (info.widgets_values) != "object") {
        //widgets_values is in some unknown inactionable format
        return
      }
      let widgetDict = info.widgets_values
      if (widgetDict.length == undefined) {
        for (let w of this.widgets) {
          if (w.name in widgetDict) {
            w.value = widgetDict[w.name];
          } else {
            //attempt to restore default value
            let inputs = LiteGraph.getNodeType(this.type).nodeData.input;
            let initialValue = null;
            if (inputs?.required?.hasOwnProperty(w.name)) {
              if (inputs.required[w.name][1]?.hasOwnProperty("default")) {
                initialValue = inputs.required[w.name][1].default;
              } else if (inputs.required[w.name][0].length) {
                initialValue = inputs.required[w.name][0][0];
              }
            } else if (inputs?.optional?.hasOwnProperty(w.name)) {
              if (inputs.optional[w.name][1]?.hasOwnProperty("default")) {
                initialValue = inputs.optional[w.name][1].default;
              } else if (inputs.optional[w.name][0].length) {
                initialValue = inputs.optional[w.name][0][0];
              }
            }
            if (initialValue) {
              w.value = initialValue;
            }
          }
        }
      } else {
        //Saved data was not a map made by this method
        //and a conversion dict for it does not exist
        //It's likely an array and that has been blindly applied
        if (info?.widgets_values?.length != this.widgets.length) {
          //Widget could not have restored properly
          //Note if multiple node loads fail, only the latest error dialog displays
          app.ui.dialog.show("Failed to restore node: " + this.title + "\nPlease remove and re-add it.")
          this.bgcolor = "#C00"
        }
      }
    });
    chainCallback(this, "onSerialize", function (info) {
      info.widgets_values = {};
      if (!this.widgets) {
        //object has no widgets, there is nothing to store
        return;
      }
      for (let w of this.widgets) {
        info.widgets_values[w.name] = w.value;
      }
    });
  })
}

function fitHeight(node) {
  node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
  node.graph.setDirtyCanvas(true);
}

function formatUploadedUrl(params) {
  if (params.url) {
    return params.url
  }

  return api.apiURL('/view?' + new URLSearchParams(params));
}

async function uploadFile(file) {
  //TODO: Add uploaded file to cache with Cache.put()?
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData();
    const i = file.webkitRelativePath.lastIndexOf('/');
    const subfolder = file.webkitRelativePath.slice(0, i + 1)
    const new_file = new File([file], file.name, {
      type: file.type,
      lastModified: file.lastModified,
    });
    body.append("image", new_file);
    if (i > 0) {
      body.append("subfolder", subfolder);
    }
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body,
    });

    if (resp.status === 200) {
      return resp.json()
    } else {
      alert(`Upload failed: ${resp.statusText}`);
    }
  } catch (error) {
    alert(`Upload failed: ${error}`);
  }
}

function addVideoCustomSize(nodeType, nodeData, widgetName) {
  //Add the extra size widgets now
  //This takes some finagling as widget order is defined by key order
  const newWidgets = {};
  for (let key in nodeData.input.required) {
    newWidgets[key] = nodeData.input.required[key]
    if (key == widgetName) {
      newWidgets[key][0] = newWidgets[key][0].concat(["Custom Width", "Custom Height", "Custom"])
      newWidgets["custom_width"] = ["INT", { "default": 512, "min": 8, "step": 8 }]
      newWidgets["custom_height"] = ["INT", { "default": 512, "min": 8, "step": 8 }]
    }
  }
  nodeData.input.required = newWidgets;

  //Add a callback which sets up the actual logic once the node is created
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const node = this;
    const sizeOptionWidget = node.widgets.find((w) => w.name === widgetName);
    const widthWidget = node.widgets.find((w) => w.name === "custom_width");
    const heightWidget = node.widgets.find((w) => w.name === "custom_height");
    injectHidden(widthWidget);
    widthWidget.options.serialize = false;
    injectHidden(heightWidget);
    heightWidget.options.serialize = false;
    sizeOptionWidget._value = sizeOptionWidget.value;
    Object.defineProperty(sizeOptionWidget, "value", {
      set: function (value) {
        //TODO: Only modify hidden/reset size when a change occurs
        if (value == "Custom Width") {
          widthWidget.hidden = false;
          heightWidget.hidden = true;
        } else if (value == "Custom Height") {
          widthWidget.hidden = true;
          heightWidget.hidden = false;
        } else if (value == "Custom") {
          widthWidget.hidden = false;
          heightWidget.hidden = false;
        } else {
          widthWidget.hidden = true;
          heightWidget.hidden = true;
        }
        node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
        this._value = value;
      },
      get: function () {
        return this._value;
      }
    });
    //Ensure proper visibility/size state for initial value
    sizeOptionWidget.value = sizeOptionWidget._value;

    sizeOptionWidget.serializeValue = function () {
      if (this.value == "Custom Width") {
        return widthWidget.value + "x?";
      } else if (this.value == "Custom Height") {
        return "?x" + heightWidget;
      } else if (this.value == "Custom") {
        return widthWidget.value + "x" + heightWidget.value;
      } else {
        return this.value;
      }
    };
  });
}

function addUploadWidget(nodeType, widgetName, type) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const pathWidget = this.widgets.find((w) => w.name === widgetName);
    const fileInput = document.createElement("input");
    chainCallback(this, "onRemoved", () => {
      fileInput?.remove();
    });

    if (type === "image") {
      Object.assign(fileInput, {
        type: "file",
        accept: "image/png,image/jpeg,image/webp",
        style: "display: none",
        multiple: true,
        onchange: async () => {
          if (!fileInput.files.length) {
            return
          }

          let successes = [];
          for (const file of fileInput.files) {
            const params = await uploadFile(file)

            if (!!params) {
              successes.push(params);
            } else {
              // Upload failed, but some prior uploads may have succeeded
              // Stop future uploads to prevent cascading failures
              // and only add to list if an upload has succeeded
              if (successes.length) {
                break
              } else {
                return;
              }
            }
          }

          pathWidget.value = successes.map(formatUploadedUrl).join("\n");
        },
      });
    } else if (type === "video") {
      Object.assign(fileInput, {
        type: "file",
        accept: "video/webm,video/mp4,video/mkv,image/gif,image/webp",
        style: "display: none",
        onchange: async () => {
          if (fileInput.files.length) {
            const params = await uploadFile(fileInput.files[0])
            if (!params) {
              // upload failed and file can not be added to options
              return;
            }

            pathWidget.value = formatUploadedUrl(params);
          }
        },
      });
    } else {
      throw new Error(`Unknown upload type ${type}`)
    }

    document.body.append(fileInput);
    let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
      //clear the active click event
      app.canvas.node_widget = null
      fileInput.click();
    });
    uploadWidget.options.serialize = false;
  });
}

function addVideoPreview(nodeType) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    let previewNode = this;
    //preview is a made up widget type to enable user defined functions
    //videopreview is widget name
    //The previous implementation used type to distinguish between a video and gif,
    //but the type is not serialized and would not survive a reload
    var previewWidget = {
      name: "videopreview", type: "preview", value: "",
      draw: function (ctx, node, widgetWidth, widgetY, height) {
        //update widget position, hide if off-screen
        const transform = ctx.getTransform();
        const scale = app.canvas.ds.scale;//gets the litegraph zoom
        //calculate coordinates with account for browser zoom
        const x = transform.e * scale / transform.a;
        const y = transform.f * scale / transform.a;
        Object.assign(this.parentEl.style, {
          left: (x + 15 * scale) + "px",
          top: (y + widgetY * scale) + "px",
          width: ((widgetWidth - 30) * scale) + "px",
          zIndex: 2 + (node.is_selected ? 1 : 0),
          position: "absolute",
        });
        this._boundingCount = 0;
      },
      computeSize: function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
          let height = (previewNode.size[0] - 30) / this.aspectRatio;
          if (!(height > 0)) {
            height = 0;
          }
          return [width, height];
        }
        return [width, -4];//no loaded src, widget should not display
      },
      _value: { hidden: false, paused: false }
    };
    //onRemoved isn't a litegraph supported function on widgets
    //Given that onremoved widget and node callbacks are sparse, this
    //saves the required iteration.
    chainCallback(this, "onRemoved", () => {
      previewWidget?.parentEl?.remove();
    });
    this.addCustomWidget(previewWidget);
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "vhs_preview";
    previewWidget.parentEl.style['pointer-events'] = "none"

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = false;
    previewWidget.videoEl.loop = true;
    previewWidget.videoEl.muted = true;
    previewWidget.videoEl.style['width'] = "100%"
    previewWidget.videoEl.addEventListener("loadedmetadata", () => {

      previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
      fitHeight(this);
    });
    previewWidget.videoEl.addEventListener("error", () => {
      //TODO: consider a way to properly notify the user why a preview isn't shown.
      previewWidget.parentEl.hidden = true;
      fitHeight(this);
    });

    previewWidget.imgEl = document.createElement("img");
    previewWidget.imgEl.style['width'] = "100%"
    previewWidget.imgEl.hidden = true;
    previewWidget.imgEl.onload = () => {
      previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
      fitHeight(this);
    };

    this.setPreviewsrc = (params) => { previewWidget._value.params = params; this._setPreviewsrc(params) };
    this._setPreviewsrc = function (params) {
      if (params == undefined) {
        return
      }
      previewWidget.parentEl.hidden = previewWidget._value.hidden;
      if (params?.format?.split('/')[0] == 'video') {
        previewWidget.videoEl.autoplay = !previewWidget._value.paused && !previewWidget._value.hidden;
        previewWidget.videoEl.src = formatUploadedUrl(params);
        previewWidget.videoEl.hidden = false;
        previewWidget.imgEl.hidden = true;
      } else {
        // Is animated image
        previewWidget.imgEl.src = formatUploadedUrl(params);
        previewWidget.videoEl.hidden = true;
        previewWidget.imgEl.hidden = false;
      }
    }
    Object.defineProperty(previewWidget, "value", {
      set: (value) => {
        if (value) {
          previewWidget._value = value
          this._setPreviewsrc(value.params)
        }
      },
      get: () => {
        return previewWidget._value;
      }
    });
    //Hide video element if offscreen
    //The multiline input implementation moves offscreen every frame
    //and doesn't apply until a node with an actual inputEl is loaded
    this._boundingCount = 0;
    this.onBounding = function () {
      if (this._boundingCount++ > 5) {
        previewWidget.parentEl.style.left = "-8000px";
      }
    }
    previewWidget.parentEl.appendChild(previewWidget.videoEl)
    previewWidget.parentEl.appendChild(previewWidget.imgEl)
    document.body.appendChild(previewWidget.parentEl);
  });

  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const pathWidget = this.widgets.find((w) => w.name === "video");
    pathWidget._value = pathWidget.value;
    Object.defineProperty(pathWidget, "value", {
      set: (value) => {
        pathWidget._value = value;
        if (!value || !/^(https?:\/\/|\/view\?)/.test(value)) {
          return
        }

        if (value.startsWith('http')) {
          return this.setPreviewsrc({ url: value });
        }

        const url = new URL(window.location.origin + value)
        const filename = url.searchParams.get("filename") || url.searchParams.get("name")
        const ext = filename.split('.').pop();
        const format = ["gif", "webp", "avif"].includes(ext) ? "image" : "video"
        this.setPreviewsrc({ filename, type: 'input', format });
      },
      get: () => {
        return pathWidget._value;
      }
    });
    //Set value to ensure preview displays on initial add.
    pathWidget.value = pathWidget._value;
  });
}

function addVideoPreviewOptions(nodeType) {
  chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
    // The intended way of appending options is returning a list of extra options,
    // but this isn't used in widgetInputs.js and would require
    // less generalization of chainCallback
    let optNew = []
    const previewWidget = this.widgets.find((w) => w.name === "videopreview");

    let url = null
    if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
      url = previewWidget.videoEl.src;
    } else if (previewWidget.imgEl?.hidden == false && previewWidget.imgEl.src) {
      url = previewWidget.imgEl.src;
    }
    if (url) {
      url = new URL(url);
      //placeholder from Save Image, will matter once preview functionality is implemented
      //url.searchParams.delete('preview')
      optNew.push(
        {
          content: "Open preview",
          callback: () => {
            window.open(url, "_blank")
          },
        },
        {
          content: "Save preview",
          callback: () => {
            const a = document.createElement("a");
            a.href = url;
            a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
            document.body.append(a);
            a.click();
            requestAnimationFrame(() => a.remove());
          },
        }
      );
    }
    const PauseDesc = (previewWidget._value.paused ? "Resume" : "Pause") + " preview";
    if (previewWidget.videoEl.hidden == false) {
      optNew.push({
        content: PauseDesc, callback: () => {
          //animated images can't be paused and are more likely to cause performance issues.
          //changing src to a single keyframe is possible,
          //For now, the option is disabled if an animated image is being displayed
          if (previewWidget._value.paused) {
            previewWidget.videoEl?.play();
          } else {
            previewWidget.videoEl?.pause();
          }
          previewWidget._value.paused = !previewWidget._value.paused;
        }
      });
    }
    //TODO: Consider hiding elements if video no preview is available yet.
    //It would reduce confusion at the cost of functionality
    //(if a video preview lags the computer, the user should be able to hide in advance)
    const visDesc = (previewWidget._value.hidden ? "Show" : "Hide") + " preview";
    optNew.push({
      content: visDesc, callback: () => {
        if (!previewWidget.videoEl.hidden && !previewWidget._value.hidden) {
          previewWidget.videoEl.pause();
        } else if (previewWidget._value.hidden && !previewWidget.videoEl.hidden && !previewWidget._value.paused) {
          previewWidget.videoEl.play();
        }
        previewWidget._value.hidden = !previewWidget._value.hidden;
        previewWidget.parentEl.hidden = previewWidget._value.hidden;
        fitHeight(this);

      }
    });
    optNew.push({
      content: "Sync preview", callback: () => {
        //TODO: address case where videos have varying length
        //Consider a system of sync groups which are opt-in?
        for (let p of document.getElementsByClassName("vhs_preview")) {
          for (let child of p.children) {
            if (child.tagName == "VIDEO") {
              child.currentTime = 0;
            } else if (child.tagName == "IMG") {
              child.src = child.src;
            }
          }
        }
      }
    });
    if (options.length > 0 && options[0] != null && optNew.length > 0) {
      optNew.push(null);
    }
    options.unshift(...optNew);
  });
}

app.registerExtension({
  name: "ArtVenture.Upload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData) return
    if (!supportedNodes.includes(nodeData?.name)) {
      return
    }

    useKVState(nodeType);

    if (nodeData.name == "LoadImageFromUrl") {
      addUploadWidget(nodeType, "url", "image");
    } else if (nodeData.name == "LoadVideoFromUrl") {
      addVideoCustomSize(nodeType, nodeData, "force_size")
      addUploadWidget(nodeType, "video", "video");
      addVideoPreview(nodeType);
      addVideoPreviewOptions(nodeType);
    }
  }
});
