#########################
#Approach #1 
#########################

# Ensure the dplyr package is loaded
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)

# Create a data frame
data <- data.frame(Letters = c("A", "B", "C", "A", "B", "C", "A", "B"))
data
# Use filter() to select A and B
selected_AB <- filter(data, Letters %in% c("A", "B"))
print("Filtered A and B:")
print(selected_AB)

# Use filter() to select either A or B
selected_AorB <- filter(data, Letters == "A" | Letters == "B")
print("Filtered A or B:")
print(selected_AorB)

#########################
#Approach #2
#########################

# Ensure the dplyr package is loaded
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)

# Create a data frame with an additional numeric column
data <- data.frame(
  Letters = c("A", "B", "C", "A", "B", "C", "A", "B"),
  Scores = c(10, 15, 10, 20, 15, 20, 5, 25)
)

# Filter to select A and B with score condition
selected_AB_Scores = filter(data, Letters %in% c("A", "B") & Scores > 10)
print("Filtered A and B with Scores > 10:")
print(selected_AB_Scores)

# Filter to select either A or B, regardless of score
selected_AorB = filter(data, Letters == "A" | Letters == "B")
print("Filtered A or B, regardless of score:")
print(selected_AorB)

