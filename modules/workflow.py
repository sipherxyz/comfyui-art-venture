import os
import re
import json
import random
import inspect
import hashlib
from datetime import datetime
from typing import Dict, List

from server import PromptServer
from folder_paths import models_dir, get_filename_list, get_full_path

from .logger import logger
from .nodes import NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS

node_class_mappings_loaded = False
ALL_NODE_CLASS_MAPPINGS = {**_NODE_CLASS_MAPPINGS}

Graph = Dict[str, List[str]]

root_dir = os.path.dirname(inspect.getfile(PromptServer))
workflows_dir = os.path.join(root_dir, "pysssss-workflows")

virtual_nodes = {"Reroute"}
input_nodes = {"AV_Input"}

checkpoint_checksums_map: Dict[str, Dict[str, str]] = {}
checkpoint_args = {"ckpt_name", "model_hash", "checkpoint"}

embeddings = set([e.split(".")[0].lower() for e in get_filename_list("embeddings")])
promp_args = {"prompt", "negative_prompt"}

seed_args = {"seed", "noise_seed"}


def get_node_class_mapping():
    global node_class_mappings_loaded

    if not node_class_mappings_loaded:
        from nodes import NODE_CLASS_MAPPINGS

        ALL_NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
        node_class_mappings_loaded = True

    return ALL_NODE_CLASS_MAPPINGS


def __dfs_sort_helper(
    graph: Graph, v: str, n: int, visited: Dict[str, bool], topNums: Dict[str, int]
) -> int:
    visited[v] = True
    neighbors = graph[v]
    for neighbor in neighbors:
        if not visited.get(neighbor, False):
            n = __dfs_sort_helper(graph, neighbor, n, visited, topNums)
    topNums[v] = n
    return n - 1


def dfs_sort(graph: Graph) -> List[str]:
    """Returns a list of vertices in their topological numbers.

    Parameters:
        graph (Graph): The graph to sort. A Graph is a dictionary of vertices and their neighbors.

    Returns:
        List[str]: Returns a list of vertices in their topological numbers.
    """

    vertices = list(graph.keys())
    visited = {}
    topNums = {}
    n = len(vertices) - 1
    for v in vertices:
        if not visited.get(v):
            n = __dfs_sort_helper(graph, v, n, visited, topNums)

    ordered_vertices = []
    for k, v in sorted(topNums.items(), key=lambda item: item[1]):
        ordered_vertices.append(k)

    return ordered_vertices


def is_seed_widget(node, widget):
    if node["type"] == "KSampler" and widget == "seed":
        return True

    if node["type"] == "KSamplerAdvanced" and widget == "noise_seed":
        return True

    return False


def load_workflow(id: str):
    workflow_path = os.path.join(workflows_dir, id + ".json")
    if os.path.isfile(workflow_path):
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    else:
        try:
            workflow = json.loads(id)
        except:
            workflow = None

    if workflow is None:
        return None

    return workflow


def map_embeddings_to_prompt(prompt: str):
    words = re.split("[,.;\s]+", prompt)
    words = [word.strip() for word in words if word.strip() != ""]

    mapped_words = [
        f"embedding:{word.lower()}" if word.lower() in embeddings else word
        for word in words
    ]

    return ", ".join(mapped_words)


def update_checkpoints_hash():
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    json_file_path = os.path.join(checkpoint_dir, "checksums.json")

    # Load existing checksum data from JSON file if it exists
    existing_checksums = {}
    if os.path.exists(json_file_path):
        with open(json_file_path) as f:
            existing_checksums = json.load(f)

    # Calculate checksum for each file in the folder
    new_checksums = {}
    for checkpoint in get_filename_list("checkpoints"):
        file_path = get_full_path("checkpoints", checkpoint)

        # Get the last modified date of the file
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

        # Check if the file is new or modified
        if (
            checkpoint not in existing_checksums
            or last_modified
            > datetime.fromisoformat(existing_checksums[checkpoint]["last_modified"])
        ):
            logger.debug(f"Calculating checksum for {checkpoint}... ")
            # Calculate the SHA256 checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            sha256_checksum = sha256_hash.hexdigest()
            # Store the new checksum and last modified date
            new_checksums[checkpoint] = {
                "shasum": sha256_checksum,
                "last_modified": last_modified.isoformat(),
            }
        else:
            # Use the existing checksum and last modified date
            new_checksums[checkpoint] = existing_checksums[checkpoint]

    # Save the new checksums to the JSON file
    with open(json_file_path, "w") as f:
        json.dump(new_checksums, f, indent=2)

    checkpoint_checksums_map.clear()
    checkpoint_checksums_map.update(new_checksums)


def get_checkpoint_by_hash(hash: str):
    for k, v in checkpoint_checksums_map.items():
        if (hash in k) or (hash in v["shasum"]):
            return k


def workflow_to_prompt(workflow, args: dict = {}):
    logger.debug("parsing workflow", json.dumps(workflow))

    graph: Graph = {}
    nodes = {node["id"]: node for node in workflow.get("nodes", [])}
    links = {}
    for link in workflow["links"]:
        [link_id, from_node, from_port, to_node, *_] = link
        from_node = from_node
        to_node = to_node
        if from_node not in graph:
            graph[from_node] = []
        if to_node not in graph:
            graph[to_node] = []

        graph[from_node].append(to_node)
        links[link_id] = link

    sorted_nodes = dfs_sort(graph)

    # build reroute map
    reroutes = {}
    for node_id in sorted_nodes:
        node = nodes[node_id]

        if (
            node["type"] in virtual_nodes
            and isinstance(node["inputs"], list)
            and len(node["inputs"]) > 0
        ):
            input = node["inputs"][0]
            link = links.get(input["link"], None)
            if not link:
                logger.error(f"Unknown link 1 {input['link']}")
                continue
            [link_id, from_node, from_port, *_] = link
            reroutes.update({node_id: (from_node, from_port)})

    # apply primitive node value
    av_input_nodes = {}
    for node_id in sorted_nodes:
        node = nodes[node_id]
        if node["type"] not in input_nodes:
            continue

        if len(node["widgets_values"]) > 1:
            arg_name = node["widgets_values"][-1]
        else:
            arg_name = node["outputs"][0]["widget"]["name"]

        def get_links(n):
            output = n.get("outputs", [{}])[0]
            _links = []
            for link_id in output.get("links", []):
                link = links.get(link_id, None)
                if not link:
                    logger.error(f"Unknown link 2 {link_id}")
                    continue
                [link_id, from_node, from_port, to_node, *_] = link
                target = nodes.get(to_node, None)
                if target is None:
                    logger.error(f"Unknown node {to_node}")
                    continue
                if target["type"] in virtual_nodes:
                    _links.extend(get_links(target))
                else:
                    _links.append(link)

            return _links

        _links = get_links(node)
        for link in _links:
            [link_id, from_node, from_port, to_node, to_port, *_] = link
            target = nodes.get(to_node, None)
            input = target.get("inputs")[to_port]
            input["value"] = node["widgets_values"][0]
            av_input_nodes[arg_name] = (str(to_node), input["name"])

    # build prompt
    prompt = {}
    for node_id in sorted_nodes:
        node = nodes[node_id]
        logger.debug(f"node {node_id:03} {node['type']} mode {node.get('mode', 0)}")

        if node.get("mode", 0) == 2:  # muted node
            continue

        if node["type"] in virtual_nodes or node["type"] in input_nodes:
            continue

        obj_class = get_node_class_mapping().get(node["type"], None)
        if obj_class is None:
            logger.error(f"Unknown node {node['type']}")
            continue

        prompt_inputs = {}
        input_def = obj_class.INPUT_TYPES()

        # handle input links
        for input in node.get("inputs", []):
            link_id = input["link"]
            if not link_id:
                continue
            link = links.get(link_id, None)
            if not link:
                logger.error(f"Unknown link 3 {link_id}")
                continue
            [link_id, from_node, from_port, *_] = link
            while from_node in reroutes:
                (from_node, from_port) = reroutes[from_node]

            prompt_inputs[input["name"]] = [str(from_node), from_port]

        # handle widget inputs
        widget_idx = 0
        for k, v in input_def.get("required", {}).items():
            widget_value = None
            v = list(v)
            if (len(v) == 1 and isinstance(v[0], list)) or len(v) == 2:
                input = next(
                    (i for i in node.get("inputs", []) if i["name"] == k), None
                )
                if input is not None:  # widget is converted to input
                    widget_value = input.get("value", None)
                if widget_value is None:
                    widget_value = node.get("widgets_values", [])[widget_idx]

                logger.debug(
                    f"  {k}: {widget_value} {'(converted to input)' if input else ''}"
                )
                prompt_inputs[k] = widget_value
                widget_idx += 1
                if k in seed_args:
                    widget_value = node.get("widgets_values", [])[widget_idx]
                    logger.debug(f"  control_after_generate: {widget_value}")
                    widget_idx += 1

        prompt[str(node_id)] = {
            "inputs": prompt_inputs,
            "class_type": node["type"],
        }

    # override args
    for k, v in av_input_nodes.items():
        node_id, input_name = v

        if k in args:
            value = args[k]
            if k in promp_args and isinstance(value, str):
                value = map_embeddings_to_prompt(value)
            elif k in checkpoint_args and isinstance(value, str):
                checkpoint = get_checkpoint_by_hash(value)
                if not checkpoint:
                    logger.error(f"Not found checkpoint {value}")
                    continue
                value = checkpoint

            logger.debug(f"arg value {k}: {value}")
            prompt[node_id]["inputs"][input_name] = value

        # random seed
        if k in seed_args:
            seed = int(prompt[node_id]["inputs"][input_name])
            if seed == 0:
                seed = random.randint(1, 1125899906842624)
                logger.debug(f"override seed value {k}: {seed}")
                prompt[node_id]["inputs"][input_name] = seed

    logger.debug("parsed prompt", json.dumps(prompt))
    return prompt
