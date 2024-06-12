interpolate_attendance = False

ratio_evaluation = [1.0, 0.75, 0.5, 0.25]
analysis_repeat = 10
relative_class = False
houldout_test_size = 1 / 3

n_best_features = [5, 10]

attendance_sheet_name = "Presencas"
grade_sheet_name = "Notas"
activity_sheet_name = "Importantes"
feeling_sheet_name = "Sentimentos"

sheets = {
    "turma_1": ".",
    "turma_2": ".",
}

hard_to_detect = [
    # turma 1
    # turma 2
    "118",  # hard
    "169",  # hard
]

dropouts = [
    # turma 1
    "Aluno 1",
    "Aluno 2",
    "Aluno 3",
    "Aluno 5",
    "Aluno 8",
    "Aluno 9",
    "Aluno 12",
    "Aluno 13",
    "Aluno 18",
    "Aluno 21",
    "Aluno 24",
    "Aluno 27",
    "Aluno 28",
    "Aluno 35",
    "Aluno 36",
    "Aluno 38",
    "Aluno 39",
    "Aluno 40",
    "Aluno 42",
    "Aluno 47",
    "Aluno 56",
    "Aluno 59",
    "Aluno 61",
    "Aluno 66",
    "Aluno 70",
    "Aluno 72",
    "Aluno 78",
    "Aluno 83",
    # turma 2
    "84",
    "100",
    "104",
    "105",
    "108",
    "120",
    "124",
    "136",
    "138",
    "144",
    "154",
    "164",
    "165",
    "171",
    "176",
]

dropouts += hard_to_detect
