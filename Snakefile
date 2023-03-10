rule make_betamax_csv_file:
     input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
     output:
        "src/data/beta_max_3_lambda_over_d.csv"
     conda:
        "environment.yml"
     script:
        "src/scripts/make_beta_max_3_lambda_over_d_csv.py"


rule make_iwa_files:
    input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
    output:
        "iwa.npy","betamin.npy","betamax.npy"
    cache:
        True
    conda:
        "environment.yml"
    script:
        "src/scripts/run_simulation.py iwa"

rule make_iwa3_files:
    input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
    output:
        "iwa3.npy","betamin3.npy","betamax3.npy"
    cache:
        True
    conda:
        "environment.yml"
    script:
        "src/scripts/run_simulation.py iwa3"