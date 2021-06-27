##*************************************************************************##
##                     Step1. mapping using BWA                            ##          
##*************************************************************************##

#!/bin/bash

for i in $(cat $input_list)
do
echo "**mapping start at $(date)**"
$bwa mem -t 4 -R $Tag $REF $in_dir/$i*R1* $in_dir/$i*R2* | $samtools view -Sb - > $out_dir/$i.bam
echo "**mapping finished at $(date)**"

##*************************************************************************##
##                 Step2. sort the bam using samtools                      ##          
##*************************************************************************##

echo "**samtools sort start at $(date)**"
$samtools sort $out_dir/$i.bam -O bam -o $out_dir/$i.sorted.bam
rm -f $out_dir/$i.bam
$samtools index $out_dir/$i.sorted.bam
echo "**samtools sort finished at $(date)**"

##*************************************************************************##
##                 Step3. calling variants using bcftools                  ##          
##*************************************************************************##
echo "**bcftools calling start at $(date)**"
$bcftools mpileup -f $REF $out_dir/$i.sorted.bam | $bcftools call -vm -Oz > $variants_out/$i.vcf.gz
echo "**bcftools calling finished at $(date)**"
done


##*************************************************************************##
##                        Step3. SNP_filter                                ##          
##*************************************************************************##
for i in $(cat $input_list)
do
$vcftools --gzvcf $variants_out/$i.vcf.gz --minDP 4 --max-missing 0.2 --minQ 30 --recode --recode-INFO-all --out $variants_out/01filter/$i.filter
### SNPs
$vcftools --gzvcf $variants_out/01filter/$i.filter.recode.vcf --remove-indels --recode --recode-INFO-all --out $variants_out/01filter/$i.filter.snps
### Indels
$vcftools --gzvcf $variants_out/01filter/$i.filter.recode.vcf --keep-only-indels --recode --recode-INFO-all --out $variants_out/01filter/$i.filter.indels

### extracting GT
$bcftools query -f '%CHROM %POS %REF %ALT [%TGT]\n' $variants_out/01filter/$i.filter.snps.recode.vcf -o $variants_out/01filter/$i.filter.snps.extract.txt
$bcftools query -f '%CHROM %POS %REF %ALT [%TGT]\n' $variants_out/01filter/$i.filter.indels.recode.vcf -o $variants_out/01filter/$i.filter.indels.extract.txt
done

## Position, Reference and Alternate
for i in $(cat $input_list); do cut -d " " -f 2,3,4 $i.filter.snps.extract.txt >$i.snp; done


