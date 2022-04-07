library(ggpubr)
library(rstatix)
library(lme4)
library(lmerTest)
library(emmeans)
library(RColorBrewer)
library(plyr)

source('../utils/helpers.R')

powerBands <- read.csv('../bandpower_with_baseline.csv')


# anova
res.aov <- anova_test(data=powerBands, dv = bandpower, wid=subjects, 
                      within=c(band, state, rounds, side)) %>%
                      get_anova_table()


# post hoc tests of changes from baseline
post_hoc <- powerBands %>% filter(!(subjects=='ocdbd2' & rounds==4)) %>%
            group_by(band) %>%
            pairwise_t_test(bandpower ~ state, paired=T, ref.group='baseline',
                            comparisons = list(c("baseline", "obsessions"), c('baseline','compulsions'),
                                               c("baseline", "relief"))) %>%
            adjust_pvalue(method="bonferroni") %>% add_significance("p.adj")



# plot significant interaction from anova with 95% confidence intervals
dataSummary <- summarySEwithin(data=powerBands, measurevar = 'bandpower',
                               withinvars = c('band', 'state'),
                               idvar = 'subjects')
ymin <- 0
ymax <- 1.5e-5

ggplot(dataSummary, aes(x=state, y=bandpower, group=band)) +
    geom_line(aes(color=band)) +
    geom_point(shape=21, size=2, fill='white') +
    geom_errorbar(width=0.1, aes(ymin=bandpower - se, ymax=bandpower+se, color=band)) +
    ylim(ymin, ymax)



