"""Generate both LiteGraph/UI and /prompt examples from the node widget schema."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("flashvsr_plus_plugin")
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__,package)
nodes = importlib.import_module("flashvsr_plus_plugin.nodes")


def definitions(cls,video=False):
    schema = cls.INPUT_TYPES()
    if cls is nodes.FlashVSRPlusUpscale and video:
        schema["optional"]["video"] = ("VIDEO",)
    widgets,inputs = {},[]
    for section in ("required","optional"):
        for name,spec in schema.get(section,{}).items():
            kind,settings = spec[0],spec[1] if len(spec)>1 else {}
            if (isinstance(kind,list) or kind in ("INT","FLOAT","BOOLEAN","STRING")) and not settings.get("forceInput"):
                widgets[name] = settings.get("default",kind[0] if isinstance(kind,list) else None)
            else:
                inputs.append({"name":name,"type":kind,"link":None})
    return widgets,inputs


def custom(node_id,cls,x,y,video=False,**overrides):
    widgets,inputs = definitions(cls,video)
    if not overrides.keys() <= widgets.keys():
        raise ValueError("Workflow overrides must name existing widgets")
    widgets.update(overrides)
    values = []
    named = dict(widgets)
    for name,value in widgets.items():
        values.append(value)
        if name == "seed":
            values.append("fixed")
            named["control_after_generate"] = "fixed"
    outputs = [{"name":name,"type":kind,"links":None,"slot_index":index}
               for index,(name,kind) in enumerate(zip(getattr(cls,"RETURN_NAMES",cls.RETURN_TYPES),cls.RETURN_TYPES))]
    return {"id":node_id,"type":cls.__name__,"pos":[x,y],"size":[340,max(120,90+len(values)*26)],
            "flags":{},"order":node_id-1,"mode":0,"inputs":inputs,"outputs":outputs,
            "properties":{"Node name for S&R":cls.__name__},"widgets_values":values,
            "widgets_values_named":named}


def standard(node_id,kind,x,y,inputs,outputs,widgets):
    return {"id":node_id,"type":kind,"pos":[x,y],"size":[290,160],"flags":{},"order":node_id-1,"mode":0,
            "inputs":[{"name":name,"type":typ,"link":None} for name,typ in inputs],
            "outputs":[{"name":name,"type":typ,"links":None,"slot_index":index}
                       for index,(name,typ) in enumerate(outputs)],
            "properties":{"Node name for S&R":kind},"widgets_values":widgets}


def connect(graph,source,slot,target,name):
    a = next(n for n in graph["nodes"] if n["id"] == source)
    b = next(n for n in graph["nodes"] if n["id"] == target)
    target_slot = next(i for i,item in enumerate(b["inputs"]) if item["name"] == name)
    link_id = len(graph["links"])+1
    a["outputs"][slot]["links"] = (a["outputs"][slot]["links"] or []) + [link_id]
    b["inputs"][target_slot]["link"] = link_id
    graph["links"].append([link_id,source,slot,target,target_slot,a["outputs"][slot]["type"]])
    graph["last_link_id"] = link_id


def graph():
    return {"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],
            "config":{},"extra":{"ds":{"scale":.7,"offset":[20,20]}},"version":.4}


def api_custom(cls,**inputs):
    widgets,_ = definitions(cls,video=True)
    return {"class_type":cls.__name__,"inputs":{**widgets,**inputs}}


def main():
    directory = ROOT / "example_workflows"
    directory.mkdir(exist_ok=True)
    image = graph()
    image["nodes"] = [
        standard(1,"LoadImage",0,30,[],[("IMAGE","IMAGE"),("MASK","MASK")],["example.png","image"]),
        custom(2,nodes.FlashVSRPlusLoader,0,250),
        custom(3,nodes.FlashVSRPlusUpscale,420,50),
        standard(4,"SaveImage",840,50,[("images","IMAGE")],[],["FlashVSR_Plus/image"]),
    ]
    connect(image,1,0,3,"images")
    connect(image,2,0,3,"pipe")
    connect(image,3,0,4,"images")
    image_api = {
        "1":{"class_type":"LoadImage","inputs":{"image":"example.png"}},
        "2":api_custom(nodes.FlashVSRPlusLoader),
        "3":api_custom(nodes.FlashVSRPlusUpscale,pipe=["2",0],images=["1",0]),
        "4":{"class_type":"SaveImage","inputs":{"images":["3",0],"filename_prefix":"FlashVSR_Plus/image"}},
    }
    video = graph()
    video["nodes"] = [
        standard(1,"LoadVideo",0,30,[],[("VIDEO","VIDEO")],["example.mp4"]),
        custom(2,nodes.FlashVSRPlusLoader,0,240),
        custom(3,nodes.FlashVSRPlusUpscale,410,30,video=True),
        custom(4,nodes.FlashVSRPlusToVideo,820,30),
        standard(5,"SaveVideo",1210,30,[("video","VIDEO")],[],["FlashVSR_Plus/video","auto","auto"]),
    ]
    connect(video,1,0,3,"video")
    connect(video,2,0,3,"pipe")
    connect(video,3,0,4,"images")
    connect(video,3,1,4,"audio")
    connect(video,3,2,4,"fps")
    connect(video,4,0,5,"video")
    video_api = {
        "1":{"class_type":"LoadVideo","inputs":{"file":"example.mp4"}},
        "2":api_custom(nodes.FlashVSRPlusLoader),
        "3":api_custom(nodes.FlashVSRPlusUpscale,pipe=["2",0],video=["1",0]),
        "4":api_custom(nodes.FlashVSRPlusToVideo,images=["3",0],audio=["3",1],fps=["3",2]),
        "5":{"class_type":"SaveVideo","inputs":{"video":["4",0],"filename_prefix":"FlashVSR_Plus/video",
                                                       "format":"auto","codec":"auto"}},
    }
    streaming = graph()
    streaming["nodes"] = [
        standard(1,"LoadVideo",0,30,[],[("VIDEO","VIDEO")],["example.mp4"]),
        custom(2,nodes.FlashVSRPlusLoader,0,240),
        custom(3,nodes.FlashVSRPlusVideoUpscale,420,30),
        standard(4,"SaveVideo",850,30,[("video","VIDEO")],[],["FlashVSR_Plus/streaming","auto","auto"]),
    ]
    connect(streaming,1,0,3,"video")
    connect(streaming,2,0,3,"pipe")
    connect(streaming,3,0,4,"video")
    streaming_api = {
        "1":{"class_type":"LoadVideo","inputs":{"file":"example.mp4"}},
        "2":api_custom(nodes.FlashVSRPlusLoader),
        "3":api_custom(nodes.FlashVSRPlusVideoUpscale,pipe=["2",0],video=["1",0]),
        "4":{"class_type":"SaveVideo","inputs":{"video":["3",0],"filename_prefix":"FlashVSR_Plus/streaming",
                                                       "format":"auto","codec":"auto"}},
    }
    reuse = copy.deepcopy(image)
    reuse["nodes"].extend([
        standard(5,"LoadImage",0,950,[],[("IMAGE","IMAGE"),("MASK","MASK")],["example_2.png","image"]),
        custom(6,nodes.FlashVSRPlusUpscale,420,950),
        standard(7,"SaveImage",840,950,[("images","IMAGE")],[],["FlashVSR_Plus/second_clip"]),
    ])
    connect(reuse,2,0,6,"pipe")
    connect(reuse,5,0,6,"images")
    connect(reuse,6,0,7,"images")
    balanced = copy.deepcopy(streaming)
    balanced["nodes"][1] = custom(2,nodes.FlashVSRPlusLoader,0,240,profile="balanced")
    # Keep existing links and port metadata from the auto example.
    balanced["nodes"][1]["outputs"] = copy.deepcopy(streaming["nodes"][1]["outputs"])
    balanced_api = copy.deepcopy(streaming_api)
    balanced_api["2"]["inputs"]["profile"] = "balanced"
    for name,data in (("image_batch",image),("native_video",video),("pipe_reuse",reuse),
                      ("streaming_video",streaming),("image_batch_api",image_api),
                      ("native_video_api",video_api),("streaming_video_api",streaming_api),
                      ("streaming_video_balanced",balanced),("streaming_video_balanced_api",balanced_api)):
        if "nodes" in data:
            data["last_node_id"] = max(n["id"] for n in data["nodes"])
        (directory / (name+".json")).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",
                                              encoding="utf-8", newline="\n")
    print("Generated 5 UI workflows and 4 API prompts.")


if __name__ == "__main__":
    main()
