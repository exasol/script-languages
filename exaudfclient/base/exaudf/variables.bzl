BENCHMARK_VM_ENABLED_DEFINE=select({
        "//exaudf:benchmark": ["ENABLE_BENCHMARK_VM"],
        "//conditions:default": []
    }) 
STREAMING_VM_ENABLED_DEFINE=select({
        "//exaudf:bash": ["ENABLE_STREAMING_VM"],
        "//conditions:default": []
    }) 
PYTHON_VM_ENABLED_DEFINE=select({
        "//exaudf:python": ["ENABLE_PYTHON_VM"],
        "//conditions:default": []
    })
JAVA_VM_ENABLED_DEFINE=select({
        "//exaudf:java": ["ENABLE_JAVA_VM"],
        "//conditions:default": []
    })

VM_ENABLED_DEFINES=BENCHMARK_VM_ENABLED_DEFINE+PYTHON_VM_ENABLED_DEFINE+JAVA_VM_ENABLED_DEFINE+STREAMING_VM_ENABLED_DEFINE
