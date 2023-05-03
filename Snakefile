rule make_phi_max_csv_file:
     input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
     output:
        "src/data/phi_max_3_lambda_over_d.csv"
     conda:
        "environment.yml"
     script:
         "src/scripts/create-data-for-circular-orbits.py"

rule make_iwa_files:
    input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
    output:
        "src/data/iwa_all.npz"
    cache:
        True
    params:
        whichsim="iwa"
    conda:
        "environment.yml"
    script:
        "src/scripts/create-data-for-elliptical-orbits.py"

rule make_iwa3_files:
    input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
    output:
        "src/data/iwa_all3.npz"
    cache:
        True
    params:
        whichsim="iwa3"
    conda:
        "environment.yml"
    script:
        "src/scripts/create-data-for-elliptical-orbits.py"