from importlib import import_module
from ._version import __version__

# Set functions to be available directly from upper scanpro, i.e. "from scanpro import scanpro"
global_classes = ["scanpro.scanpro.scanpro",
                  "scanpro.scanpro.run_scanpro",
                  "scanpro.scanpro.anova",
                  "scanpro.scanpro.t_test",
                  "scanpro.scanpro.sim_scanpro"
                  ]

for c in global_classes:

    module_name = ".".join(c.split(".")[:-1])
    attribute_name = c.split(".")[-1]

    module = import_module(module_name)
    attribute = getattr(module, attribute_name)

    globals()[attribute_name] = attribute
