##*********************Step1: merge SNPs of all samples***************************##
library("data.table")
list <- read.table("snp.list",stringsAsFactors=F)
list_vec = list$V1
sample_num = length(list_vec)

## first_sample
merge_data = fread(list_vec[1])
names(merge_data)=c("V1","REF","ALT1")

## merge all samples
for (i in seq_along(list_vec[2:sample_num])){
  new_data = fread(list_vec[2:sample_num][[i]])
  merge_data = merge(merge_data,new_data,by="V1",all=T)
  names(merge_data)[names(merge_data)=="V2"] = "ref_temp"
  names(merge_data)[names(merge_data)=="V3"] = "ALT"
  merge_data[which(is.na(merge_data$REF)),][,2]=merge_data[which(is.na(merge_data$REF)),][,"ref_temp"]
  merge_data = merge_data[,-"ref_temp"]
  names(merge_data)[names(merge_data)=="ALT"] = list_vec[2:sample_num][[i]]
 }
 
## label encoding 
merge_data[is.na(merge_data)] = "N"
merge_data[merge_data =="A"]=1
merge_data[merge_data=="G"]=2
merge_data[merge_data=="C"]=3
merge_data[merge_data=="T"]=4
merge_data[merge_data=="N"]=0

names(merge_data)[1]="POS"
merge_data = as.data.frame(merge_data)
merge_data2 <- as.data.frame(lapply(merge_data,as.numeric))

tinput <- as.data.frame(t(merge_data2))
row.names(tinput)[3]=list_vec[1]
dim(tinput)

##*****************Step2: calculate the number of 0 for each column*******************##
f <- function(x)
 sum(x==0)
tinput2 = tinput[3:dim(tinput)[1],]
num_0 <-t(as.data.frame(apply(tinput2,2,f)))
tinput3 <- as.data.frame(t(rbind(num_0,tinput)))
colnames(tinput3)[1]="Num"

##*****************Step3: Retained variants loci were present in >20% of the samples*************##
value = sample_num/5*4
tinput4 <- subset(tinput3,Num < value)
tinput5 <- as.data.frame(t(tinput4))
dim(tinput5)
write.csv(tinput5,file="snp_input.csv",quote=F,row.names=F)

############################ FCGR Encoding using kaos package #####################
data <- read.csv("snp_input.csv",header=T,stringsAsFactors=F) 
data[data==0] = "N"
data[data==1] = "A"
data[data==2] = "G"
data[data==3] = "C"
data[data==4] = "T"
dim(data)

### load kaos pacakge 
library(kaos)
data2 <- as.data.frame(t(data))
data3 <- lapply(data2,as.character) 
for (i in seq(900)){
  cgr_list <- cgr(data3[[i]],seq.base=c("N","A","G","C","T"),res = 200)
  cgr_matrix <- cgr_list$matrix
  write.csv(cgr_matrix,file=list_vec[i],quote=F,row.names=F)
  }  
  
