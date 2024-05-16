import { app, ComfyApp, ANIM_PREVIEW_WIDGET } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';
import { $el } from '../../../scripts/ui.js';
import { createImageHost } from '../../../scripts/ui/imagePreview.js';

const style = `
.comfy-img-preview video {
  object-fit: contain;
  width: var(--comfy-img-preview-width);
  height: var(--comfy-img-preview-height);
}
`;

const URL_REGEX = /^((blob:)?https?:\/\/|\/view\?|data:image\/)/;

const supportedNodes = ['LoadImageFromUrl', 'LoadImageAsMaskFromUrl', 'LoadVideoFromUrl'];

function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error('Tried to add callback to non-existant object');
    return;
  }
  if (property in object) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
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
  widget._type = widget.type;
  Object.defineProperty(widget, 'type', {
    set: function (value) {
      widget._type = value;
    },
    get: function () {
      if (widget.hidden) {
        return 'hidden';
      }
      return widget._type;
    },
  });
}

function addKVState(nodeType) {
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    chainCallback(this, 'onConfigure', function (info) {
      if (!this.widgets) {
        //Node has no widgets, there is nothing to restore
        return;
      }
      if (typeof info.widgets_values != 'object') {
        //widgets_values is in some unknown inactionable format
        return;
      }
      let widgetDict = info.widgets_values;
      if (widgetDict.length == undefined) {
        for (let w of this.widgets) {
          if (w.name in widgetDict) {
            w.value = widgetDict[w.name];
          } else {
            //attempt to restore default value
            let inputs = LiteGraph.getNodeType(this.type).nodeData.input;
            let initialValue = null;
            if (inputs?.required?.hasOwnProperty(w.name)) {
              if (inputs.required[w.name][1]?.hasOwnProperty('default')) {
                initialValue = inputs.required[w.name][1].default;
              } else if (inputs.required[w.name][0].length) {
                initialValue = inputs.required[w.name][0][0];
              }
            } else if (inputs?.optional?.hasOwnProperty(w.name)) {
              if (inputs.optional[w.name][1]?.hasOwnProperty('default')) {
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
          app.ui.dialog.show(
            'Failed to restore node: ' + this.title + '\nPlease remove and re-add it.',
          );
          this.bgcolor = '#C00';
        }
      }
    });
    chainCallback(this, 'onSerialize', function (info) {
      info.widgets_values = {};
      if (!this.widgets) {
        //object has no widgets, there is nothing to store
        return;
      }
      for (let w of this.widgets) {
        info.widgets_values[w.name] = w.value;
      }
    });
  });
}

function migrateWidget(nodeType, oldWidgetName, newWidgetName) {
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    if (!this.widgets) return

    const oldIndex = this.widgets.findIndex(w => w.name === oldWidgetName)
    if (oldIndex > -1) {
      this.widgets.splice(oldIndex, 1)
    }

    chainCallback(this, 'onConfigure', function (info) {
      if (typeof info.widgets_values != 'object') return

      const newWidget = this.widgets.find(w => w.name === newWidgetName)
      if (newWidget && info.widgets_values[oldWidgetName]) {
        newWidget.value = info.widgets_values[oldWidgetName]
      }
    })
  })
}

function formatImageUrl(params) {
  if (typeof params === 'string') {
    if (URL_REGEX.test(params)) return params;

    const folder_separator = params.lastIndexOf('/');
    let subfolder = '';
    if (folder_separator > -1) {
      subfolder = params.substring(0, folder_separator);
      params = params.substring(folder_separator + 1);
    }
    return api.apiURL(
      `/view?filename=${encodeURIComponent(
        params,
      )}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`,
    );
  }

  if (params.url) {
    return params.url;
  }

  params = { ...params };

  if (!params.filename && params.name) {
    params.filename = params.name;
    delete params.name;
  }

  return api.apiURL('/view?' + new URLSearchParams(params));
}

async function uploadFile(file) {
  //TODO: Add uploaded file to cache with Cache.put()?
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData();
    const i = file.webkitRelativePath.lastIndexOf('/');
    const subfolder = file.webkitRelativePath.slice(0, i + 1);
    const new_file = new File([file], file.name, {
      type: file.type,
      lastModified: file.lastModified,
    });
    body.append('image', new_file);
    if (i > 0) {
      body.append('subfolder', subfolder);
    }
    const resp = await api.fetchApi('/upload/image', {
      method: 'POST',
      body,
    });

    if (resp.status === 200 || resp.status === 201) {
      return resp.json();
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
    newWidgets[key] = nodeData.input.required[key];
    if (key == widgetName) {
      newWidgets[key][0] = newWidgets[key][0].concat(['Custom Width', 'Custom Height', 'Custom']);
      newWidgets['custom_width'] = ['INT', { default: 512, min: 8, step: 8 }];
      newWidgets['custom_height'] = ['INT', { default: 512, min: 8, step: 8 }];
    }
  }
  nodeData.input.required = newWidgets;

  //Add a callback which sets up the actual logic once the node is created
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    const node = this;
    const sizeOptionWidget = node.widgets.find((w) => w.name === widgetName);
    const widthWidget = node.widgets.find((w) => w.name === 'custom_width');
    const heightWidget = node.widgets.find((w) => w.name === 'custom_height');
    injectHidden(widthWidget);
    widthWidget.options.serialize = false;
    injectHidden(heightWidget);
    heightWidget.options.serialize = false;
    sizeOptionWidget._value = sizeOptionWidget.value;
    Object.defineProperty(sizeOptionWidget, 'value', {
      set: function (value) {
        //TODO: Only modify hidden/reset size when a change occurs
        if (value == 'Custom Width') {
          widthWidget.hidden = false;
          heightWidget.hidden = true;
        } else if (value == 'Custom Height') {
          widthWidget.hidden = true;
          heightWidget.hidden = false;
        } else if (value == 'Custom') {
          widthWidget.hidden = false;
          heightWidget.hidden = false;
        } else {
          widthWidget.hidden = true;
          heightWidget.hidden = true;
        }
        node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
        this._value = value;
      },
      get: function () {
        return this._value;
      },
    });
    //Ensure proper visibility/size state for initial value
    sizeOptionWidget.value = sizeOptionWidget._value;

    sizeOptionWidget.serializeValue = function () {
      if (this.value == 'Custom Width') {
        return widthWidget.value + 'x?';
      } else if (this.value == 'Custom Height') {
        return '?x' + heightWidget;
      } else if (this.value == 'Custom') {
        return widthWidget.value + 'x' + heightWidget.value;
      } else {
        return this.value;
      }
    };
  });
}

function addUploadWidget(nodeType, widgetName, type) {
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    const pathWidget = this.widgets.find((w) => w.name === widgetName);
    if (pathWidget.element) {
      pathWidget.options.getMinHeight = () => 50;
      pathWidget.options.getMaxHeight = () => 150;
    }

    const fileInput = document.createElement('input');
    chainCallback(this, 'onRemoved', () => {
      fileInput?.remove();
    });

    if (type === 'image') {
      Object.assign(fileInput, {
        type: 'file',
        accept: 'image/png,image/jpeg,image/webp',
        style: 'display: none',
        multiple: true,
        onchange: async () => {
          if (!fileInput.files.length) {
            return;
          }

          let successes = [];
          for (const file of fileInput.files) {
            const params = await uploadFile(file);

            if (!!params) {
              successes.push(params);
            } else {
              // Upload failed, but some prior uploads may have succeeded
              // Stop future uploads to prevent cascading failures
              // and only add to list if an upload has succeeded
              if (successes.length) {
                break;
              } else {
                return;
              }
            }
          }

          pathWidget.callback(successes.map(formatImageUrl).join('\n'));
          fileInput.value = '';
        },
      });
    } else if (type === 'video') {
      Object.assign(fileInput, {
        type: 'file',
        accept: 'video/webm,video/mp4,video/mkv,image/gif,image/webp',
        style: 'display: none',
        multiple: true,
        onchange: async () => {
          if (!fileInput.files.length) {
            return;
          }

          let successes = [];
          for (const file of fileInput.files) {
            const params = await uploadFile(file);

            if (!!params) {
              successes.push(params);
            } else {
              // Upload failed, but some prior uploads may have succeeded
              // Stop future uploads to prevent cascading failures
              // and only add to list if an upload has succeeded
              if (successes.length) {
                break;
              } else {
                return;
              }
            }
          }

          pathWidget.callback(successes.map(formatImageUrl).join('\n'));
          fileInput.value = '';
        },
      });
    } else {
      throw new Error(`Unknown upload type ${type}`);
    }

    document.body.append(fileInput);
    let uploadWidget = this.addWidget('button', 'choose ' + type + ' to upload', 'image', () => {
      //clear the active click event
      app.canvas.node_widget = null;
      fileInput.click();
    });
    uploadWidget.options.serialize = false;
  });
}

function patchValueSetter(nodeType, widgetName) {
  chainCallback(nodeType.prototype, 'onNodeCreated', function () {
    const pathWidget = this.widgets.find(w => w.name === widgetName)
    pathWidget._value = pathWidget.value
    let editing = false

    const setter = value => {
      pathWidget._value = value
      this.images = (value ?? "").split("\n")
      if (pathWidget.type === "customtext" && !editing) {
        pathWidget.inputEl.value = value
      }
    }

    Object.defineProperty(pathWidget, "value", {
      set: setter,
      get: () => pathWidget._value,
    })

    if (pathWidget.type === "customtext") {
      pathWidget.inputEl.addEventListener("focus", (e) => {
        editing = true
      })
      pathWidget.inputEl.addEventListener("blur", (e) => {
        editing = false
      })
      pathWidget.inputEl.addEventListener("keyup", (e) => {
        setter(e.target.value)
      })
    }

    pathWidget.callback = setter
    pathWidget.value = pathWidget._value
  });
}

function addVideoPreview(nodeType, widgetName) {
  const createVideoNode = (url) => {
    return new Promise((cb) => {
      const videoEl = document.createElement('video');
      Object.defineProperty(videoEl, 'naturalWidth', {
        get: () => {
          return videoEl.videoWidth;
        },
      });
      Object.defineProperty(videoEl, 'naturalHeight', {
        get: () => {
          return videoEl.videoHeight;
        },
      });
      videoEl.addEventListener('loadedmetadata', () => {
        videoEl.controls = false;
        videoEl.loop = true;
        videoEl.muted = true;
        cb(videoEl);
      });
      videoEl.addEventListener('error', () => {
        cb();
      });
      videoEl.src = url;
    });
  };

  const createImageNode = (url) => {
    return new Promise((cb) => {
      const imgEl = document.createElement('img');
      imgEl.onload = () => {
        cb(imgEl);
      };
      imgEl.addEventListener('error', () => {
        cb();
      });
      imgEl.src = url;
    });
  };

  nodeType.prototype.onDrawBackground = function (ctx) {
    if (this.flags.collapsed) return;

    let imageURLs = (this.images ?? []).map(formatImageUrl);
    let imagesChanged = false;

    if (JSON.stringify(this.displayingImages) !== JSON.stringify(imageURLs)) {
      this.displayingImages = imageURLs;
      imagesChanged = true;
    }

    if (!imagesChanged) return;
    if (!imageURLs.length) {
      this.imgs = null;
      this.animatedImages = false;
      return;
    }

    const promises = imageURLs.map((url) => {
      if (url.startsWith('/view')) {
        url = window.location.origin + url;
      }

      const u = new URL(url);
      const filename =
        u.searchParams.get('filename') || u.searchParams.get('name') || u.pathname.split('/').pop();
      const ext = filename.split('.').pop();
      const format = ['gif', 'webp', 'avif'].includes(ext) ? 'image' : 'video';
      if (format === 'video') {
        return createVideoNode(url);
      } else {
        return createImageNode(url);
      }
    });

    Promise.all(promises)
      .then((imgs) => {
        this.imgs = imgs.filter(Boolean);
      })
      .then(() => {
        if (!this.imgs.length) return;

        this.animatedImages = true;
        const widgetIdx = this.widgets?.findIndex((w) => w.name === ANIM_PREVIEW_WIDGET);

        // Instead of using the canvas we'll use a IMG
        if (widgetIdx > -1) {
          // Replace content
          const widget = this.widgets[widgetIdx];
          widget.options.host.updateImages(this.imgs);
        } else {
          const host = createImageHost(this);
          this.setSizeForImage(true);
          const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', host.el, {
            host,
            getHeight: host.getHeight,
            onDraw: host.onDraw,
            hideOnZoom: false,
          });
          widget.serializeValue = () => ({
            height: host.el.clientHeight,
          });
          // widget.computeSize = (w) => ([w, 220]);

          widget.options.host.updateImages(this.imgs);
        }

        this.imgs.forEach((img) => {
          if (img instanceof HTMLVideoElement) {
            img.muted = true;
            img.autoplay = true;
            img.play();
          }
        });
      });
  };

  patchValueSetter(nodeType, widgetName)

  chainCallback(nodeType.prototype, 'onExecuted', function (message) {
    if (message?.videos) {
      this.images = message?.videos.map(formatImageUrl);
    }
  });
}

function addImagePreview(nodeType, widgetName) {
  const onDrawBackground = nodeType.prototype.onDrawBackground;
  nodeType.prototype.onDrawBackground = function (ctx) {
    if (this.flags.collapsed) return;

    let imageURLs = (this.images ?? []).map(formatImageUrl);
    let imagesChanged = false;

    if (JSON.stringify(this.displayingImages) !== JSON.stringify(imageURLs)) {
      this.displayingImages = imageURLs;
      imagesChanged = true;
    }

    if (imagesChanged) {
      this.imageIndex = null;
      if (imageURLs.length > 0) {
        Promise.all(
          imageURLs.map((src) => {
            return new Promise((r) => {
              const img = new Image();
              img.onload = () => r(img);
              img.onerror = () => r(null);
              img.src = src;
            });
          }),
        ).then((imgs) => {
          this.imgs = imgs.filter(Boolean);
          this.setSizeForImage?.();
          app.graph.setDirtyCanvas(true);
        });
      } else {
        this.imgs = null;
      }
    }

    onDrawBackground?.call(this, ctx);
  };

  patchValueSetter(nodeType, widgetName)
}

app.registerExtension({
  name: 'ArtVenture.Upload',
  init() {
    $el('style', {
      textContent: style,
      parent: document.head,
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData) return;
    if (!supportedNodes.includes(nodeData?.name)) {
      return;
    }

    addKVState(nodeType)

    if (nodeData.name === 'LoadImageFromUrl' || nodeData.name === 'LoadImageAsMaskFromUrl') {
      migrateWidget(nodeType, 'url', 'image')
      addUploadWidget(nodeType, 'image', 'image');
      addImagePreview(nodeType, 'image');
    } else if (nodeData.name == 'LoadVideoFromUrl') {
      addVideoCustomSize(nodeType, nodeData, 'force_size');
      addUploadWidget(nodeType, 'video', 'video');
      addVideoPreview(nodeType, 'video');
    }
  },
});
