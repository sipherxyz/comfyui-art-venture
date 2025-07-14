import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { $el } from '../../scripts/ui.js';
import { addWidget, DOMWidgetImpl } from '../../scripts/domWidget.js';
import { ComfyWidgets } from '../../scripts/widgets.js'

import { chainCallback, addKVState, addWidgetChangeCallback } from './utils.js';

const style = `
.comfy-img-preview video {
  object-fit: contain;
  width: var(--comfy-img-preview-width);
  height: var(--comfy-img-preview-height);
}
`;

const supportedNodes = ['LoadImageFromUrl', 'LoadImageAsMaskFromUrl'];

const formatUrl = (url) => {
  if (!url) return ""

  if (url.startsWith("http://") || url.startsWith("https://") || url.startsWith("blob:")) return url
  if (url.startsWith("/view") || url.startsWith("/api/view")) return url

  let type = "input"
  if (url.endsWith(']')) {
    const openBracketIndex = url.lastIndexOf('[')
    type = url.slice(openBracketIndex + 1, url.length - 1).trim()
    url = url.slice(0, openBracketIndex).trim()
  }

  const parts = url.split('/')
  const filename = parts.pop()
  const subfolder = parts.join('/')

  const params = [
    'filename=' + encodeURIComponent(filename),
    'type=' + type,
    'subfolder=' + subfolder,
    app.getRandParam().substring(1)
  ].join('&')

  return api.apiURL(`/view?${params}`)
}

// copied from ComfyUI_frontend/src/composables/widgets/useStringWidget.ts
// remove the Object.defineProperty(widget, 'value') part
function addUrlWidget(node, name, options) {
  const inputEl = document.createElement('textarea')
  inputEl.className = 'comfy-multiline-input'
  inputEl.value = options.default
  inputEl.placeholder = options.placeholder || name
  inputEl.spellcheck = false

  const widget = new DOMWidgetImpl({
    node,
    name,
    type: 'customtext',
    element: inputEl,
    options: {
      hideOnZoom: true,
      getValue() {
        return inputEl.value
      },
      setValue(v) {
        inputEl.value = v
      }
    }
  })
  addWidget(node, widget)

  widget.inputEl = inputEl
  widget.options.minNodeSize = [400, 200]

  inputEl.addEventListener('input', () => {
    widget.value = inputEl.value
    widget.callback?.(inputEl.value, true)
  })

  // Allow middle mouse button panning
  inputEl.addEventListener('pointerdown', (event) => {
    if (event.button === 1) {
      app.canvas.processMouseDown(event)
    }
  })

  inputEl.addEventListener('pointermove', (event) => {
    if ((event.buttons & 4) === 4) {
      app.canvas.processMouseMove(event)
    }
  })

  inputEl.addEventListener('pointerup', (event) => {
    if (event.button === 1) {
      app.canvas.processMouseUp(event)
    }
  })

  /** Timer reference. `null` when the timer completes. */
  let ignoreEventsTimer = null
  /** Total number of events ignored since the timer started. */
  let ignoredEvents = 0

  // Pass wheel events to the canvas when appropriate
  inputEl.addEventListener('wheel', (event) => {
    if (!Object.is(event.deltaX, -0)) return

    // If the textarea has focus, require more effort to activate pass-through
    const multiplier = document.activeElement === inputEl ? 2 : 1
    const maxScrollHeight = inputEl.scrollHeight - inputEl.clientHeight

    if (
      (event.deltaY < 0 && inputEl.scrollTop === 0) ||
      (event.deltaY > 0 && inputEl.scrollTop === maxScrollHeight)
    ) {
      // Attempting to scroll past the end of the textarea
      if (!ignoreEventsTimer || ignoredEvents > 25 * multiplier) {
        app.canvas.processMouseWheel(event)
      } else {
        ignoredEvents++
      }
    } else if (event.deltaY !== 0) {
      // Start timer whenever a successful scroll occurs
      ignoredEvents = 0
      if (ignoreEventsTimer) clearTimeout(ignoreEventsTimer)

      ignoreEventsTimer = setTimeout(() => {
        ignoreEventsTimer = null
      }, 800 * multiplier)
    }
  })

  return widget
}

function addImageUploadWidget(nodeType, nodeData, imageInputName) {
  const { input } = nodeData ?? {}
  const required = input?.required
  if (!required) return

  const imageOptions = required.image
  delete required.image

  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    this.previewMediaType = 'image'

    const urlWidget = addUrlWidget(this, imageInputName, imageOptions[1])
    // move urlWidget to the first position
    const widgets = this.widgets.filter(w => w !== urlWidget)
    this.widgets = [urlWidget, ...widgets]

    ComfyWidgets.IMAGEUPLOAD(
      this,
      'upload',
      ["IMAGEUPLOAD", { "image_upload": true, imageInputName }],
    )

    const safeLoadImageFromUrl = (url) => {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(null);
        img.src = url;
      });
    }

    let initialImgs = undefined;
    let isUrlImageSet = false

    const setImagesFromUrl = (value = "") => {
      this.imageIndex = null;

      const urls = value.split("\n").filter(Boolean).map(formatUrl);
      if (!urls.length) {
        this.imgs = undefined;
        this.widgets = this.widgets.filter((w) => w.name !== "$$canvas-image-preview");
        isUrlImageSet = true;
        return
      }

      return Promise.all(
        urls.map(safeLoadImageFromUrl)
      ).then((imgs) => {
        initialImgs = imgs.filter(Boolean);
        this.imgs = initialImgs.length > 0 ? initialImgs : undefined;
        if (!this.imgs) {
          this.widgets = this.widgets.filter((w) => w.name !== "$$canvas-image-preview");
        }
        app.graph.setDirtyCanvas(true);

        // cancel any img change in the next 2 seconds
        // to prevent `ComfyUI `overwriting the image
        setTimeout(() => {
          isUrlImageSet = true;
        }, 2000);

        return initialImgs;
      })
    }

    addWidgetChangeCallback(this, {
      name: "imgs",
      shouldChange: (value) => {
        if (isUrlImageSet) return true;
        return value === initialImgs;
      }
    })

    urlWidget.callback = (value) => {
      if (!value) { // from upload
        value = urlWidget.value.split("\n").filter(Boolean).map(formatUrl).join("\n")
      }
      if (Array.isArray(value)) {
        value = value.map(formatUrl).join("\n")
      }
      if (value !== urlWidget.options.getValue()) {
        urlWidget.options.setValue(value)
      }
      setImagesFromUrl(value)
    }

    this.clipspace = () => {
      const widgets = this.widgets
        .map(({ type, name, value }) => ({
          type,
          name,
          value,
        }))

      widgets.push(
        {
          type: "text",
          name: imageInputName,
          value: (urlWidget.value || "").split("\n").filter(Boolean)[0],
        },
        {
          type: "text",
          name: "url",
          value: (urlWidget.value || "").split("\n").filter(Boolean)[0],
        }
      )

      return { widgets, images: undefined }
    }

    requestAnimationFrame(() => {
      setImagesFromUrl(urlWidget.value);
    })
  })
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

    if (nodeData.name === 'LoadImageFromUrl' || nodeData.name === 'LoadImageAsMaskFromUrl') {
      addImageUploadWidget(nodeType, nodeData, 'image');
    }

    addKVState(nodeType);
  },
});
