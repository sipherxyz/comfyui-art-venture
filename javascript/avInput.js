import { ComfyWidgets } from "/scripts/widgets.js";
import { app } from "/scripts/app.js";

function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	let linkType = type;
	if (type instanceof Array) {
		type = "COMBO";
		linkType = linkType.join(",");
	}
	return { type, linkType };
}

app.registerExtension({
	name: "AV.WidgetInputs",
	registerCustomNodes() {
		class AVInputNode {
			color=LGraphCanvas.node_colors.green.color;
            bgcolor=LGraphCanvas.node_colors.green.bgcolor;
            groupcolor = LGraphCanvas.node_colors.green.groupcolor;

			constructor() {
				this.addOutput("Connect to AV Input", "*");
				this.serialize_widgets = true;
				this.isVirtualNode = true;
			}

			applyToGraph() {
				if (!this.outputs[0].links?.length) return;

				function get_links(node) {
					let links = [];
					for (const l of node.outputs[0].links) {
						const linkInfo = app.graph.links[l];
						const n = node.graph.getNodeById(linkInfo.target_id);
						if (n.type == "Reroute") {
							links = links.concat(get_links(n));
						} else {
							links.push(l);
						}
					}
					return links;
				}

				let links = get_links(this);
				// For each output link copy our value over the original widget value
				for (const l of links) {
					const linkInfo = app.graph.links[l];
					const node = this.graph.getNodeById(linkInfo.target_id);
					const input = node.inputs[linkInfo.target_slot];
					const widgetName = input.widget.name;
					if (widgetName) {
						const widget = node.widgets.find((w) => w.name === widgetName);
						if (widget) {
							widget.value = this.widgets[0].value;
							if (widget.callback) {
								widget.callback(widget.value, app.canvas, node, app.canvas.graph_mouse, {});
							}
						}
					}
				}
			}

			onConnectionsChange(_, index, connected) {
				if (connected) {
					if (this.outputs[0].links?.length) {
						if (!this.widgets?.length) {
							this.#onFirstConnection();
						}
						if (!this.widgets?.length && this.outputs[0].widget) {
							// On first load it often cant recreate the widget as the other node doesnt exist yet
							// Manually recreate it from the output info
							this.#createWidget(this.outputs[0].widget.config);
						}
					}
				} else if (!this.outputs[0].links?.length) {
					this.#onLastDisconnect();
				} else if (this.outputs[0].links?.length === 1) {
					let linkId = this.outputs[0].links[0];
					let link = this.graph.links[linkId];
					if (!link) return this.#onLastDisconnect();

					while (link) {
						const theirNode = this.graph.getNodeById(link.target_id);
						if (theirNode.type === "Reroute" && theirNode.outputs[0].links?.length) {
							link = this.graph.links[theirNode.outputs[0].links[0]];
							if (!link) return this.#onLastDisconnect();
						} else if (theirNode.type === "Reroute") {
							// last node is a reroute with no links
							return this.#onLastDisconnect();
						} else {
							// last node is not a reroute
							break;
						}
					}
				}
			}

			onConnectOutput(slot, type, input, target_node, target_slot) {
				// Fires before the link is made allowing us to reject it if it isn't valid
				if (target_node.type === "Reroute") {
					return true;
				}

				// No widget, we cant connect
				if (!input.widget) {
					if (!(input.type in ComfyWidgets)) return false;
				}

				if (this.outputs[slot].links?.length) {
					return this.#isValidConnection(input);
				}
			}

			#onFirstConnection() {
				// First connection can fire before the graph is ready on initial load so random things can be missing
				const linkId = this.outputs[0].links[0];
				let link = this.graph.links[linkId];
				if (!link) return;

				let theirNode = this.graph.getNodeById(link.target_id);
				if (!theirNode || !theirNode.inputs) return;
				// follow Reroute
				const self = this;
				while (theirNode.type === "Reroute") {
					const onConnectionsChange = theirNode.onConnectionsChange;
					if (!onConnectionsChange || onConnectionsChange.patcher !== self) {
						theirNode.onConnectionsChange = function (...args) {
							console.log(args)
							self.onConnectionsChange.apply(self, args);
							if (onConnectionsChange) {
								onConnectionsChange.apply(theirNode, args);
							}
						};
						theirNode.onConnectionsChange.patcher = self;
					}

					if (!theirNode.outputs[0].links) {
						return;
					}

					const rerouteLinkId = theirNode.outputs[0].links[0];
					link = this.graph.links[rerouteLinkId];
					if (!link) return;
					theirNode = this.graph.getNodeById(link.target_id);
				}

				const input = theirNode.inputs[link.target_slot];
				if (!input) return;

				var _widget;
				if (!input.widget) {
					if (!(input.type in ComfyWidgets)) return;
					_widget = { "name": input.name, "config": [input.type, {}] }//fake widget
				} else {
					_widget = input.widget;
				}

				const widget = _widget;
				const { type, linkType } = getWidgetType(widget.config);
				// Update our output to restrict to the widget type
				this.outputs[0].type = linkType;
				this.outputs[0].name = type;
				this.outputs[0].widget = widget;

				this.#createWidget(widget.config, theirNode, widget.name);
			}

			#createWidget(inputData, node, widgetName) {
				let type = inputData[0];

				if (type instanceof Array) {
					type = "COMBO";
				}

				let widget;
				if (type in ComfyWidgets) {
					widget = (ComfyWidgets[type](this, widgetName || "value", inputData, app) || {}).widget;
				} else {
					widget = this.addWidget(type, widgetName || "value", null, () => { }, {});
				}

				if (node?.widgets && widget) {
					const theirWidget = node.widgets.find((w) => w.name === widgetName);
					if (theirWidget) {
						widget.value = theirWidget.value;
					}
				}

				this.addWidget("text", "name", widgetName, () => { }, {})

				// When our value changes, update other widgets to reflect our changes
				// e.g. so LoadImage shows correct image
				const callback = widget.callback;
				const self = this;
				widget.callback = function () {
					const r = callback ? callback.apply(this, arguments) : undefined;
					self.applyToGraph();
					return r;
				};

				// Grow our node if required
				const sz = this.computeSize();
				if (this.size[0] < sz[0]) {
					this.size[0] = sz[0];
				}
				if (this.size[1] < sz[1]) {
					this.size[1] = sz[1];
				}

				requestAnimationFrame(() => {
					if (this.onResize) {
						this.onResize(this.size);
					}
				});
			}

			#isValidConnection(input) {
				// Only allow connections where the configs match
				const config1 = this.outputs[0].widget.config;
				const config2 = input.widget.config;

				if (config1[0] instanceof Array) {
					// These checks shouldnt actually be necessary as the types should match
					// but double checking doesn't hurt

					// New input isnt a combo
					if (!(config2[0] instanceof Array)) return false;
					// New imput combo has a different size
					if (config1[0].length !== config2[0].length) return false;
					// New input combo has different elements
					if (config1[0].find((v, i) => config2[0][i] !== v)) return false;
				} else if (config1[0] !== config2[0]) {
					// Configs dont match
					return false;
				}

				for (const k in config1[1]) {
					if (k !== "default") {
						if (config1[1][k] !== config2[1][k]) {
							return false;
						}
					}
				}

				return true;
			}

			#onLastDisconnect() {
				// We cant remove + re-add the output here as if you drag a link over the same link
				// it removes, then re-adds, causing it to break
				this.outputs[0].type = "*";
				this.outputs[0].name = "Recipe Input";
				delete this.outputs[0].widget;

				if (this.widgets) {
					// Allow widgets to cleanup
					for (const w of this.widgets) {
						if (w.onRemove) {
							w.onRemove();
						}
					}
					this.widgets.length = 0;
				}
			}
		}

		LiteGraph.registerNodeType(
			"AV_Input",
			Object.assign(AVInputNode, {
				title: "AV Recipe Input",
			})
		);
		AVInputNode.category = "Art Venture";
	},
});
