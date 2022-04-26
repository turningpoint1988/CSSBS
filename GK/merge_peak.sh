#!/usr/bin/bash

target=${1}
# merge all peaks
cat `pwd`/${target}/GM12878_${target}_peak.bed `pwd`/${target}/K562_${target}_peak.bed > `pwd`/${target}/GK_${target}.bed
sort -k1,1 -k2,2n `pwd`/${target}/GK_${target}.bed > `pwd`/${target}/GK_sort.bed
bedtools merge -i `pwd`/${target}/GK_sort.bed > `pwd`/${target}/GK_merge.bed
rm -f `pwd`/${target}/GK_${target}.bed `pwd`/${target}/GK_sort.bed
  
# count mapping reads from two replicates
samtools sort -T ./ -o `pwd`/${target}/GM12878_bam/sorted1.bam `pwd`/${target}/GM12878_bam/GM12878_${target}_bam1.bam
samtools index `pwd`/${target}/GM12878_bam/sorted1.bam
samtools sort -T ./ -o `pwd`/${target}/GM12878_bam/sorted2.bam `pwd`/${target}/GM12878_bam/GM12878_${target}_bam2.bam
samtools index `pwd`/${target}/GM12878_bam/sorted2.bam
bedtools multicov -f 0.1 -bams `pwd`/${target}/GM12878_bam/sorted1.bam \
                               `pwd`/${target}/GM12878_bam/sorted2.bam \
                               -bed `pwd`/${target}/GK_merge.bed > `pwd`/${target}/GM12878_count.bed
#
samtools sort -T ./ -o `pwd`/${target}/K562_bam/sorted1.bam `pwd`/${target}/K562_bam/K562_${target}_bam1.bam
samtools index `pwd`/${target}/K562_bam/sorted1.bam
samtools sort -T ./ -o `pwd`/${target}/K562_bam/sorted2.bam `pwd`/${target}/K562_bam/K562_${target}_bam2.bam
samtools index `pwd`/${target}/K562_bam/sorted2.bam   
bedtools multicov -f 0.1 -bams `pwd`/${target}/K562_bam/sorted1.bam \
                               `pwd`/${target}/K562_bam/sorted2.bam \
                               -bed `pwd`/${target}/GK_merge.bed > `pwd`/${target}/K562_count.bed


