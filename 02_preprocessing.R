##*********************merge snp of all samples***************************##
list <- read.table("snp.list",stringsAsFactors=F)
list_vec = list$V1
## first_sample
merge_data = fread(list_vec[1])
names(merge_data)=c("V1","REF","ALT1")
for (i in seq_along(list_vec[2:933])){
  new_data = fread(list_vec[2:933][[i]])
  merge_data = merge(merge_data,new_data,by="V1",all=T)
  names(merge_data)[names(merge_data)=="V2"] = "ref_temp"
  names(merge_data)[names(merge_data)=="V3"] = "ALT"
  merge_data[which(is.na(merge_data$REF)),][,2]=merge_data[which(is.na(merge_data$REF)),][,"ref_temp"]
  merge_data = merge_data[,-"ref_temp"]
  names(merge_data)[names(merge_data)=="ALT"] = list_vec[2:933][[i]]
 }
 
 
merge_data[is.na(merge_data)] = "N"
merge_data[merge_data =="A"]=1
merge_data[merge_data=="G"]=2
merge_data[merge_data=="C"]=3
merge_data[merge_data=="T"]=4
merge_data[merge_data=="N"]=0

names(merge_data)[1]="POS"
merge_data = as.data.frame(merge_data)
merge_data2 <- as.data.frame(lapply(merge_data,as.numeric))
colnames(merge_data2)=colnames(merge_data)

tinput <- as.data.frame(t(merge_data2))
row.names(tinput)[3]=list_vec[1]
dim(tinput)

##*********************calculate the number of 0 for each column***************************##
f <- function(x)
 sum(x==0)
num_0 <-t(as.data.frame(apply(tinput,2,f)))

tinput2 <- as.data.frame(t(rbind(num_0,tinput2)))
colnames(tinput2)[1]="Num"

##### 200
tinput3 <- subset(tinput2,Num < 200)
tinput4 <- as.data.frame(t(tinput3))
dim(tinput4)
write.csv(tinput2,file="snp_input.csv",quote=F,row.names=F)

############################ FCGR Encoding using kaos package #####################
rawinput <- read.csv("cip_gi_200.csv",header=T,stringsAsFactors=F) 
dim(rawinput)
data <- rawinput
data[data==0] = "N"
data[data==1] = "A"
data[data==2] = "G"
data[data==3] = "C"
data[data==4] = "T"
dim(data)

### load kaos pacakge 
library(kaos)
data <- as.data.frame(data)
pheno <- read.csv("CIP_gi_pheno.csv",header=T,stringsAsFactors=F) 
list_vec = as.character(pheno$prename)
row.names(data) = pheno$prename
data2 <- as.data.frame(t(data))
data3 <- lapply(data2,as.character) 
for (i in seq(900)){
  cgr_list <- cgr(data3[[i]],seq.base=c("N","A","G","C","T"),res = 200)
  cgr_matrix <- cgr_list$matrix
  write.csv(cgr_matrix,file=list_vec[i],quote=F,row.names=F)
  }  

# combine = np.vstack(d_com_arr)
# rawdata = pd.DataFrame(combine)
# output=combine.reshape(900,200,200) 
# np.save("alt_cnn_input.npy",output)    
  

 