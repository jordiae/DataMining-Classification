data <- read.csv("../data/original/bank-additional-full.csv", header=TRUE, sep=";")

#Separation Training - testing dataset

seed <- 1234
percent_train <- 0.75

## 75% of the sample size
smp_size <- floor(percent_train * nrow(data))

## set the seed to make your partition reproducible
set.seed(seed)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]
