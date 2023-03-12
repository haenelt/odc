# -*- coding: utf-8 -*-
"""Some constants."""

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

SUBJECTS = ["p1", "p2", "p3", "p4", "p5"]
N_LAYER = 11

SESSION: dict[str, dict] = {}
SESSION["p1"] = {}
SESSION["p1"]["GE_EPI"] = [3, 4]
SESSION["p1"]["SE_EPI"] = [1, 2]
SESSION["p1"]["VASO"] = [1, 2]
SESSION["p1"]["VASO_uncorrected"] = [1, 2]

SESSION["p2"] = {}
SESSION["p2"]["GE_EPI"] = [1, 2]
SESSION["p2"]["SE_EPI"] = [1, 2]
SESSION["p2"]["VASO"] = [2, 3]
SESSION["p2"]["VASO_uncorrected"] = [2, 3]

SESSION["p3"] = {}
SESSION["p3"]["GE_EPI"] = [2, 3]
SESSION["p3"]["SE_EPI"] = [1, 2]
SESSION["p3"]["VASO"] = [1, 3]
SESSION["p3"]["VASO_uncorrected"] = [1, 3]

SESSION["p4"] = {}
SESSION["p4"]["GE_EPI"] = [1, 2]
SESSION["p4"]["SE_EPI"] = [1, 2]
SESSION["p4"]["VASO"] = [1, 2]
SESSION["p4"]["VASO_uncorrected"] = [1, 2]

SESSION["p5"] = {}
SESSION["p5"]["GE_EPI"] = [1, 2]
SESSION["p5"]["SE_EPI"] = [1, 2]
SESSION["p5"]["VASO"] = [1, 2]
SESSION["p5"]["VASO_uncorrected"] = [1, 2]
