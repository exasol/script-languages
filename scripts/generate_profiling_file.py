import re
import sys
from pathlib import Path
from typing import List, Tuple


def create_events_from_log(log_file: Path):
    events = list()
    with open(log_file) as log_f:
        for line in log_f:  # PROFILING[BEGIN addExternalJarPaths] 17:59:53.694
            event_match = re.search(r"PROFILING\[([\w, ,-]+)\] ([\d,:,.]+)", line)
            if event_match is not None:
                time_stamp_str = event_match.groups()[1]
                event_name = event_match.groups()[0]
                time_stamp_match = re.search(r"([\d]+):([\d]+):([\d]+).([\d]+)", time_stamp_str)
                if len(time_stamp_match.groups()) != 4:
                    raise RuntimeError(f"Invalid timestamp: {time_stamp_str}")
                time_stamp_parsed_millis = (int(time_stamp_match.groups()[0]) * 3600 +
                                            int(time_stamp_match.groups()[1]) * 60 +
                                            int(time_stamp_match.groups()[2])) * 1000 +\
                                            int(time_stamp_match.groups()[3])
                events.append((event_name, time_stamp_parsed_millis))
    return events


stack_graph = {"SQL QUERY": {
    "UDF MAIN": {
        "UDF-LIB MAIN": {
            "JavaVMImpl-ctor": {
                "createJvm": None,
                "compileScript": None
            },
            "JavaVM-run": {
                "ExaWrapper-run": None
            },
            "JavaVMImpl-shutdown": None
        }
    }}}


def find_event(event_list: list, event_key: str) -> Tuple[str, float]:
    for event in event_list:
        if event[0] == event_key:
            return event


def calculate_time_diff(event_list: list, start_event_key: str, end_event_key: str) -> float:
    start_event = find_event(event_list, start_event_key)
    end_event = find_event(event_list, end_event_key)
    if start_event is None:
        raise RuntimeError(f"Could not find start event: {start_event_key}")
    if end_event is None:
        raise RuntimeError(f"Could not find end event: {end_event_key}")
    time_diff = end_event[1] - start_event[1]
    return time_diff


def build_event_graph(stack_sub_graph: dict, event_list: list, event_ptr: dict):
    for key in stack_sub_graph.keys():
        v = stack_sub_graph[key]
        time = 0
        if v is None:
            entry = calculate_time_diff(event_list, f"BEGIN {key}", f"END {key}")
            event_ptr[key] = (entry, None)
        else:
            child_keys = v.keys()
            child_key_iterator = iter(child_keys)
            first_child_key = next(child_key_iterator)
            entry = calculate_time_diff(event_list, f"BEGIN {key}", f"BEGIN {first_child_key}")
            child_dict = dict()
            event_ptr[key] = (entry, child_dict)
            build_event_graph(v, event_list, child_dict)


def print_event_graph(event_graph: dict, parent: str):
    for key in event_graph.keys():
        v = event_graph[key]
        print(f"{parent}{key} {v[0]}")
        child = v[1]
        if child:
            print_event_graph(child, f"{parent}{key};")


if __name__ == "__main__":
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        raise ValueError(f"log path does not exists:{log_path}")
    events = create_events_from_log(log_path)

    event_dict = dict()
    build_event_graph(stack_graph, events, event_dict)
    print_event_graph(event_dict, "")


## You can use this script to compile an input file for:
## https://github.com/brendangregg/FlameGraph
## For example create the input file:
## python3 generate_profiling_file.py .build_output/jobs/job1/outputs/TestContainer_1/test_output > flame_chart_input.txt
## ./flamegraph.pl -flamechart -title "My title" flame_chart_input.txt > result.svg
