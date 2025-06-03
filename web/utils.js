export const CONVERTED_TYPE = "converted-widget"

export function hideWidgetForGood(node, widget, suffix = "") {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, ":" + widget.name)
    }
  }
}

const doesInputWithNameExist = (node, name) => {
  return node.inputs ? node.inputs.some(input => input.name === name) : false
}

const HIDDEN_TAG = "tschide"
const origProps = {}

// Toggle Widget + change size
export function toggleWidget(node, widget, show = false, suffix = "", updateSize = true) {
  if (!widget || doesInputWithNameExist(node, widget.name)) return

  // Store the original properties of the widget if not already stored
  if (!origProps[widget.name]) {
    origProps[widget.name] = {
      origType: widget.type,
      origComputeSize: widget.computeSize,
    }
  }

  const origSize = node.size

  // Set the widget type and computeSize based on the show flag
  widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix
  widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4]

  // Recursively handle linked widgets if they exist
  widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show))

  // Calculate the new height for the node based on its computeSize method
  if (updateSize) {
    const newHeight = node.computeSize()[1]
    node.setSize([node.size[0], newHeight])
  }
}

export function addWidgetChangeCallback(widget, callback) {
  let widgetValue = widget.value
  let originalDescriptor = Object.getOwnPropertyDescriptor(widget, "value")
  Object.defineProperty(widget, "value", {
    get() {
      return originalDescriptor && originalDescriptor.get ? originalDescriptor.get.call(widget) : widgetValue
    },
    set(newVal) {
      if (originalDescriptor && originalDescriptor.set) {
        originalDescriptor.set.call(widget, newVal)
      } else {
        widgetValue = newVal
      }

      callback(newVal)
    },
  })
}

export function chainCallback(object, property, callback) {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object")
    return
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig?.apply(this, arguments)
      callback.apply(this, arguments)
      return r
    }
  } else {
    object[property] = callback
  }
}

export function addKVState(nodeType) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    chainCallback(this, "onConfigure", function (info) {
      if (!this.widgets) {
        //Node has no widgets, there is nothing to restore
        return
      }
      if (typeof info.widgets_values != "object") {
        //widgets_values is in some unknown inactionable format
        return
      }
      let widgetDict = info.widgets_values
      if (widgetDict.length == undefined) {
        for (let w of this.widgets) {
          if (w.name in widgetDict) {
            w.value = widgetDict[w.name]
            if (w.type !== "button") {
              w.callback?.(w.value)
            }
          } else {
            //attempt to restore default value
            let inputs = LiteGraph.getNodeType(this.type).nodeData.input
            let initialValue = null
            if (inputs?.required?.hasOwnProperty(w.name)) {
              if (inputs.required[w.name][1]?.hasOwnProperty("default")) {
                initialValue = inputs.required[w.name][1].default
              } else if (inputs.required[w.name][0].length) {
                initialValue = inputs.required[w.name][0][0]
              }
            } else if (inputs?.optional?.hasOwnProperty(w.name)) {
              if (inputs.optional[w.name][1]?.hasOwnProperty("default")) {
                initialValue = inputs.optional[w.name][1].default
              } else if (inputs.optional[w.name][0].length) {
                initialValue = inputs.optional[w.name][0][0]
              }
            }
            if (initialValue) {
              w.value = initialValue
              if (w.type !== "button") {
                w.callback?.(w.value)
              }
            }
          }
        }
      }
    })
    chainCallback(this, "onSerialize", function (info) {
      info.widgets_values = {}
      if (!this.widgets) {
        //object has no widgets, there is nothing to store
        return
      }
      for (let w of this.widgets) {
        info.widgets_values[w.name] = w.value
      }
    })
  })
}
