"""Few-shot OECD 5-digit purpose coding with a hosted FLAN-UL2, packaging the 2023 method.

Modules, one job each:
    taxonomy   the official OECD purpose-code list
    prompt     one project's prompt from the sector's guidelines and k examples
    selection  random, hand-picked and similarity-based example selection
    votek      Vote-K selective annotation
    backend    the TextGenerator interface and the FLAN-UL2 SageMaker adapter
    classify   the command-line classifier
"""
