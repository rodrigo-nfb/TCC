def recall(confusion_matrix: dict):  # noqa
    if confusion_matrix["true_positive"] + confusion_matrix["false_negative"] != 0:
        return confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_negative"]
        )
    else:
        return -1


def precision(confusion_matrix: dict):  # noqa

    if confusion_matrix["true_positive"] + confusion_matrix["false_positive"] != 0:
        return confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_positive"]
        )
    else:
        return -1


def f1(confusion_matrix: dict):  # noqa
    if (precision(confusion_matrix) + recall(confusion_matrix)) != 0:
        return (
            2
            * (precision(confusion_matrix) * recall(confusion_matrix))
            / (precision(confusion_matrix) + recall(confusion_matrix))
        )
    else:
        return -1


def accuracy(confusion_matrix: dict):  # noqa
    return (confusion_matrix["true_positive"] + confusion_matrix["true_negative"]) / (
        confusion_matrix["true_positive"]
        + confusion_matrix["true_negative"]
        + confusion_matrix["false_positive"]
        + confusion_matrix["false_negative"]
    )
