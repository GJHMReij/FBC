# Inlezen van Cohort_alles_selectie.xlsx in R

# Eenmalig installeren indien nodig:
# install.packages("readxl")

library(readxl)

# Pas dit pad aan naar de locatie van jouw bestand
pad <- "Cohort_alles_selectie.xlsx"

# Bekijk welke tabbladen (sheets) het bestand bevat
excel_sheets(pad)

# Eerste sheet inlezen
cohort <- read_excel(pad, sheet = 1)

# Snelle inspectie
str(cohort)
head(cohort)

# Als je een specifiek tabblad wilt inlezen, gebruik de naam of het nummer:
# cohort <- read_excel(pad, sheet = "NaamVanTabblad")

# Als de header niet in de eerste rij staat, bijv. rij 3:
# cohort <- read_excel(pad, sheet = 1, skip = 2)
