nextflow.enable.dsl=2
params.epochs = params.epochs ?: null

workflow {

    raw_data_ch = Channel
        .fromPath(params.input_dirs.collect { it + "/*.csv" })
        .map { f ->
            // If input_dirs are .../Numerical_data_MAR_1/data, dataset is the parent of "data"
            def dataset = f.parent?.parent?.name ?: "unknown_dataset"
            tuple(f, dataset)
        }

    method_imp_ch = Channel
        .fromList(params.imputation_settings.entrySet())
        .flatMap { entry ->
            def method = entry.key
            def count  = entry.value as int
            return (1..count).collect { imp -> tuple(method, imp) }
        }

    jobs_ch = raw_data_ch.combine(method_imp_ch)
    impute(jobs_ch)
}


process impute {

    tag "${csv_file.name} | ${method} | imp${imp}"
    container 'serology-imputation:latest'

    publishDir { "${params.output_root}/${dataset}/${method}" }, mode: 'copy', overwrite: true

    input:
    tuple path(csv_file), val(dataset), val(method), val(imp)

    output:
    path "${csv_file.baseName}_imp${imp}_imputed.csv"

    script:
    def epoch_flag = params.epochs ? "--epochs ${params.epochs}" : ""
    def expected_out = "${csv_file.baseName}_imp${imp}_imputed.csv"

    """
    serology-impute \
      --file ${csv_file} \
      --method ${method} \
      --seed ${imp} \
      ${epoch_flag} \
      --out-dir .

    # No mv needed — Nextflow will collect the uniquely named output file
    ls -lh ${expected_out}
    """
}
