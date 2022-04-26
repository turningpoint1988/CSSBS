# Library
library(DESeq2)

setwd("~/CSSBS/GK")
target <- 'CTCF'
# input comparative results
GM12878 <- read.table(paste(target, 'GM12878_count.bed', sep = '/'), header = FALSE)
# RPKM = mapped reads of a target peak / (total mapped reads of all peaks*length of this target peak) * 1e09
total_reads_1 <- sum(GM12878$V4)
total_reads_2 <- sum(GM12878$V5)

GM12878_name <- c()
for (i in 1:dim(GM12878)[1]) {
  len <- GM12878[i,3] - GM12878[i,2]
  rpkm1 <- GM12878[i,4] / (total_reads_1 * len) * 1e09
  rpkm2 <- GM12878[i,5] / (total_reads_2 * len) * 1e09
  GM12878[i,4] <- rpkm1
  GM12878[i,5] <- rpkm2
  name <- paste(paste(GM12878[i,1], GM12878[i,2], sep = ":"), GM12878[i,3], sep = "-")
  GM12878_name <- c(GM12878_name, name)
}
GM12878[is.na(GM12878)] <- 0
matrix_G <- cbind(GM12878_name, GM12878$V4, GM12878$V5)
rownames(matrix_G) <- matrix_G[,1]
matrix_G <- matrix_G[,-1]
colnames(matrix_G) <- c("control1", "control2")
#
K562 <- read.table(paste(target, 'K562_count.bed', sep = '/'), header = FALSE)
# RPKM = mapped reads of a target peak / (total mapped reads of all peaks*length of this target peak) * 1e09
total_reads_1 <- sum(K562$V4)
total_reads_2 <- sum(K562$V5)
K562_name <- c()
for (i in 1:dim(K562)[1]) {
  len <- K562[i,3] - K562[i,2]
  rpkm1 <- K562[i,4] / (total_reads_1 * len) * 1e09
  rpkm2 <- K562[i,5] / (total_reads_2 * len) * 1e09
  K562[i,4] <- rpkm1
  K562[i,5] <- rpkm2
  name <- paste(paste(K562[i,1], K562[i,2], sep = ":"), K562[i,3], sep = "-")
  K562_name <- c(K562_name, name)
}
K562[is.na(K562)] <- 0
matrix_K <- cbind(K562_name, K562$V4, K562$V5)
rownames(matrix_K) <- matrix_K[,1]
matrix_K <- matrix_K[,-1]
colnames(matrix_K) <- c("treat1", "treat2")

matrix_diff <- cbind(matrix_G, matrix_K)
matrix_diff <- apply(matrix_diff, 2, as.integer)

index <- which(apply(matrix_diff, 1, sum) > 60)
matrix_diff <- matrix_diff[index,]
filtered_name <- K562_name[index]
matrix_diff <- as.data.frame(matrix_diff, row.names = filtered_name)
#
condition <- factor(c(rep("control",2),rep("treat",2)), levels = c("control","treat"))
colData <- data.frame(row.names=colnames(matrix_diff), condition)
# 
input <- DESeqDataSetFromMatrix(matrix_diff, colData, design= ~ condition)
output <- DESeq(input)
res= results(output)
res = res[order(res$pvalue),]
head(res)
# select
diff_plus <- subset(res, padj < 0.05 & log2FoldChange > 2)
diff_minus <- subset(res, padj < 0.05 & log2FoldChange < -2)
similarity <- subset(res, padj >= 0.1 & abs(log2FoldChange) <= 1)
#
beds_plus <- c()
for (name in diff_plus@rownames) {
  name_split <- strsplit(name,"[:-]")[[1]]
  beds_plus <- rbind(beds_plus, name_split)
}
write.table(beds_plus, file= paste(target, 'K562_specific.txt', sep = '/'), quote = FALSE, row.names = FALSE, col.names = FALSE, sep = '\t')
beds_minus <- c()
for (name in diff_minus@rownames) {
  name_split <- strsplit(name,"[:-]")[[1]]
  beds_minus <- rbind(beds_minus, name_split)
}
write.table(beds_minus, file= paste(target, 'GM12878_specific.txt', sep = '/'), quote = FALSE, row.names = FALSE, col.names = FALSE, sep = '\t')
beds_sim <- c()
for (name in similarity@rownames) {
  name_split <- strsplit(name,"[:-]")[[1]]
  beds_sim <- rbind(beds_sim, name_split)
}
write.table(beds_sim, file= paste(target, 'Shared.txt', sep = '/'), quote = FALSE, row.names = FALSE, col.names = FALSE, sep = '\t')
