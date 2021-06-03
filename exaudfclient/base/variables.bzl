BENCHMARK_VM_ENABLED_DEFINE=select({
        "//:benchmark": ["ENABLE_BENCHMARK_VM"],
        "//conditions:default": []
    }) 
CPP_EMIT_BENCHMARK_VM_ENABLED_DEFINE=select({
        "//:cpp-emit-benchmark": ["ENABLE_CPP_EMIT_BENCHMARK_VM"],
        "//conditions:default": []
    }) 
STREAMING_VM_ENABLED_DEFINE=select({
        "//:bash": ["ENABLE_STREAMING_VM"],
        "//conditions:default": []
    }) 
PYTHON_VM_ENABLED_DEFINE=select({
        "//:python": ["ENABLE_PYTHON_VM"],
        "//conditions:default": []
    })
JAVA_VM_ENABLED_DEFINE=select({
        "//:java": ["ENABLE_JAVA_VM"],
        "//conditions:default": []
    })

VM_ENABLED_DEFINES=CPP_EMIT_BENCHMARK_VM_ENABLED_DEFINE+BENCHMARK_VM_ENABLED_DEFINE+PYTHON_VM_ENABLED_DEFINE+JAVA_VM_ENABLED_DEFINE+STREAMING_VM_ENABLED_DEFINE
