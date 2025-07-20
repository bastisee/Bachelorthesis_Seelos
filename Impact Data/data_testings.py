import pandas as pd

# Test Series Throw
throw_points = {
    "Point": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
    "Coordinates (x, y)": [(30, 110), (90, 110), (150, 110),
                           (30, 80), (90, 80), (150, 80),
                           (30, 40), (90, 40), (150, 40)]
}
df_throw = pd.DataFrame(throw_points)
print("Test Series Throw:\n", df_throw, "\n")

# Stage 1 Shots
stage1 = {
    "Number": ["#2", "#5", "#7", "#8", "#9", "#10", "#11"],
    "real Coordinates (x,y)": [(80, 50), (76, 38), (65, 51),
                               (100, 34), (119, 50), (108, 68), (134, 72)]
}
df_stage1 = pd.DataFrame(stage1)
print("Stage1_Shots:\n", df_stage1, "\n")

# Stage 2 & 3 Shots
stage2_3 = {
    "Number": ["#1", "#2", "#3", "#8", "#9", "#10", "#11",
               "#13", "#15", "#17", "#19", "#20"],
    "real Coordinates (x,y)": [(27, 64), (79, 45), (81, 19),
                               (132, 45), (86, 87), (56, 16), (104, 44),
                               (97, 70), (89, 77), (114, 42), (78, 42), (123, 29)]
}
df_stage2_3 = pd.DataFrame(stage2_3)
print("Stage2_3_Shots:\n", df_stage2_3, "\n")

# Stage 4 Shots
stage4 = {
    "Number": ["#1", "#3", "#4", "#5", "#6", "#7", "#8",
               "#9", "#11", "#14", "#15", "#16"],
    "real Coordinates (x,y)": [(63, 44), (114, 66), (59, 36),
                               (88, 70), (151, 60), (175, 55), (54, 52),
                               (60, 50), (60, 67), (118, 89), (114, 65), (39, 71)]
}
df_stage4 = pd.DataFrame(stage4)
print("Stage4_Shots:\n", df_stage4)
