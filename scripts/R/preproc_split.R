library(caret)
path <- "../../plots/"
data <- read.csv("../../data/original/bank-additional-full.csv", header=TRUE, sep=";")
#Creation previously contacted
summary(data)
data_proc <- data
data_proc$prev_contacted <- 1
data_proc$prev_contacted[data_proc$pdays == 999] <- 0
data_proc$prev_contacted <- factor(data_proc$prev_contacted,
                                   levels = c(1,0),
                                   labels = c("yes", "no"))
#Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
data_proc$duration <- NULL

#Splitting:
seed <- 1234
percent_train <- 0.70
set.seed(seed)

#per defecte fa un stratified
train.index <- createDataPartition(data_proc$y, p = percent_train, list = FALSE)
train <- data_proc[ train.index,]
test  <- data_proc[-train.index,]

#Separem els Y i els N per poder fer el test
data_proc_y = data_proc[data_proc$y == 'yes',]
data_proc_n = data_proc[data_proc$y == 'no',]
train_y = train[train$y == 'yes',]
train_n = train[train$y == 'no',]
test_y = test[test$y == 'yes',]
test_n = test[test$y == 'no',]
#Check p-value of numeric values
contador <- 0
for(column in names(data_proc)){
  if(is.numeric(data_proc[,column])){
    res <- ks.test(train_y[,column], data_proc_y[,column])
    res2 <- ks.test(train_n[,column], data_proc_n[,column])
    res3 <- ks.test(test_y[,column], data_proc_y[,column])
    res4 <- ks.test(train_n[,column], data_proc_n[,column])
    if(res$p.value <= 0.5 || res4$p.value <= 0.5 || res3$p.value <= 0.5 || res2$p.value <= 0.5) break;
  }else{
    print(column)
    png(paste(path,column,"_histogram_proc_y.png",sep = ""))
    plot(histogram(data_proc_y[,column]))
    dev.off()
    png(paste(path,column,"_histogram_train_y.png",sep = ""))
    plot(histogram(train_y[,column]))
    dev.off()
    png(paste(path,column,"_histogram_test_y.png",sep = ""))
    plot(histogram(test_y[,column]))
    dev.off()
    png(paste(path,column,"_histogram_proc_n.png",sep = ""))
    plot(histogram(data_proc_n[,column]))
    dev.off()
    png(paste(path,column,"_histogram_train_n.png",sep = ""))
    plot(histogram(train_n[,column]))
    dev.off()
    png(paste(path,column,"_histogram_test_n.png",sep = ""))
    plot(histogram(test_n[,column]))
    dev.off()
  }
  contador <- contador + 1
}
if(contador != length(names(data_proc))) print("error!");
write.table(test,file ="../data/BankCleanTest.csv",row.names=FALSE,col.names=TRUE,sep=";")
write.table(train,file = "../data/BankCleanTrain.csv",row.names=FALSE,col.names=TRUE,sep=";")

#done
