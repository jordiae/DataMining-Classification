data <- read.csv("../data/original/bank-additional-full.csv", header=TRUE, sep=";")
#Creation previously contacted
summary(data)
data_proc <- data
data_proc$prev_contacted <- 'yes'
data_proc$prev_contacted[data_proc$pdays == 999] <- 'no'
#Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
data_proc$duration <- NULL