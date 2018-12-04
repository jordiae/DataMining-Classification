data <- read.csv(file ="bank-additional-full.txt" ,header = TRUE, sep = ";")
install.packages("dplyr")
library("dplyr")



#Find the unknown values
sapply(data,function(x) sum(x == "unknown"))
sapply(data,function(x) sum(x == "999"))
sapply(data,function(x) sum(x == "0"))
class(data)
dim(data)
n<-dim(data)[1]

K<-dim(data)[2]
k<-names(data)




par(ask=TRUE)

for(k in 1:K){
  if (is.factor(data[,k])){ 
    frecs<-table(data[,k], useNA="ifany")
    proportions<-frecs/n
    #ojo, decidir si calcular porcentages con o sin missing values
    pie(frecs, cex=0.6, main=paste("Pie of", names(data)[k]))
    barplot(frecs, las=3, cex.names=0.7, main=paste("Barplot of", names(data)[k]))
    print(frecs)
    print(proportions)
  }else{
    hist( data[,k], main=paste("Histogram of", names(data)[k]))
    boxplot( data[,k], horizontal=TRUE, main=paste("Boxplot of", names(data)[k]))
    print(summary(data[,k]))
    print(paste("sd: ", sd(data[,k])))
    print(paste("vc: ", sd(data[,k])/mean(data[,k])))
  }
}


hist( data$campaign, breaks = seq(0, 56, by = 1))
summary(data$campaign)
summary(data$pdays)
datapdays <- filter(data, data$pdays != 999)
hist(datapdays$pdays)
summary(data$age)




#Bivariate Analysis
#creating categories for age
counts <- table(data$y,data$day_of_week)
barplot(counts,,
        xlab="Day of the week",
        legend = rownames(counts), beside=TRUE)
data
counts <- table(data$y,data$age)
barplot(counts,,
        xlab="Day of the week",
        legend = rownames(counts), beside=TRUE)



     

