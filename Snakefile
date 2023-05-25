rule make_phi_max_csv_file:
     input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
     output:
        "src/data/phi_max_3_lambda_over_d.csv"
     conda:
        "environment.yml"
     script:
         "src/scripts/create-data-for-circular-orbits.py"

rule create_eccentric_orbits_data:
    input:
        "src/data/2646_NASA_ExEP_Target_List_HWO_Table.csv"
    output:
        "src/data/eccentric-orbits.npz"
    cache:
        True
    conda:
        "environment.yml"
    script:
        "src/scripts/create-data-for-eccentric-orbits.py"
