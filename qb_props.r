# QB PROPS MODEL!!!!!
library(data.table)
library(DT)
library(dplyr)
library(caret)
# used for training - predicted variables!!! 
weekly_pass_raw = rbind(data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2018.csv"))%>% select(-opponent_team),
                 data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2019.csv")),
                 data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2020.csv"))%>% select(-opponent_team),
                 data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2021.csv")),
                 data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2022.csv")),
                 data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2023.csv")) %>% select(-opponent_team))
weekly_pass = weekly_pass_raw %>%
  filter(position == "QB") %>%
  select(player_id,player_display_name,recent_team,season,week,completions,attempts,passing_yards)%>%
  rename(team = "recent_team")

season_pass = weekly_pass_raw %>%
  filter(position == "QB")%>%
  group_by(player_display_name,player_id,season,recent_team,position)%>%
  summarize(games = n(),
            completions_per_game = mean(completions,na.rm=TRUE), 
            sd_completions_per_game = sd(completions,na.rm=TRUE), 
            completions_per_attempt = sum(completions) / sum(attempts,na.rm=TRUE),
            attempts_per_game = mean(attempts,na.rm=TRUE),
            sd_attempts_per_game = sd(attempts,na.rm=TRUE),
            sacks_per_game_per_game = mean(sacks,na.rm=TRUE),
            sd_sacks_per_game_per_game = sd(sacks,na.rm=TRUE),
            sack_yards_per_game = mean(sack_yards,na.rm=TRUE),
            sack_fumples_per_game = mean(sack_fumbles,na.rm=TRUE),
            passing_air_yards_per_game = mean(passing_air_yards,na.rm=TRUE),
            sd_passing_air_yards_per_game = sd(passing_air_yards,na.rm=TRUE),
            passing_air_yards_per_attempt = sum(passing_air_yards,na.rm=TRUE) / sum(attempts,na.rm=TRUE),
            passing_yards_after_catch_per_game = mean(passing_yards_after_catch,na.rm=TRUE),
            sd_passing_yards_after_catch_per_game = sd(passing_yards_after_catch,na.rm=TRUE),
            passing_yards_after_catch_per_attempt = sum(passing_yards_after_catch,na.rm=TRUE) / sum(attempts,na.rm=TRUE),
            passing_first_downs_per_game = mean(passing_first_downs,na.rm=TRUE),
            passing_epa_per_game = mean(passing_epa,na.rm=TRUE),
            sd_passing_epa_per_game = sd(passing_epa,na.rm=TRUE),
            passing_2pt_conversions_per_game = mean(passing_2pt_conversions,na.rm=TRUE),
            avg_pacr = mean(pacr,na.rm=TRUE),
            var_pacr = var(pacr,na.rm=TRUE),
            avg_dakota = mean(dakota,na.rm=TRUE),
            var_dakota = var(dakota,na.rm=TRUE),
            carries_per_game = mean(carries,na.rm=TRUE),
            sd_carries_per_game = sd(carries,na.rm=TRUE),
            rushing_yards_per_game = mean(rushing_yards,na.rm=TRUE),
            sd_rushing_yards_per_game = sd(rushing_yards,na.rm=TRUE))%>%
  rename(team = "recent_team")
colnames(season_pass) = paste0("QB_SZN_",colnames(season_pass))
            
# season stats - need to pull based on week 
adv_rec = data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_season_rec.csv"))%>%
  group_by(player,pfr_id,season,tm,pos)%>%
  summarize(season_rec_targets_p_game = as.numeric(tgt / g),
            season_rec_p_game = as.numeric(rec / g),
            season_rec_p_target = as.numeric(rec / tgt),
            season_rec_yards_p_game = as.numeric(yds / g),
            season_ybc_p_r = as.numeric(ybc_r),
            season_yac_p_r = as.numeric(yac_r),
            season_broken_tackle_p_game = as.numeric(brk_tkl / g),
            season_drop_percent = as.numeric(drop_percent),
            season_int_p_target = as.numeric(int/tgt),
            season_adot = as.numeric(adot))

# adv_rush = data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_season_rush.csv"))%>%
#   group_by(player,pfr_id,season,tm,pos)%>%
#   summarize(season_rush_attempts_p_game = att/g,
#             season_rush_yards_p_game = yds/g,
#             season_rush_yards_before_contact_p_attempt = ybc_att,
#             season_rush_yards_after_contact_p_attempt = yac_att,
#             season_rush_attempts_p_broken_tackle = att_br)
adv_offense = adv_rec %>% 
  as.data.frame() %>%
  select(-player)#merge(adv_rec,adv_rush,by=c("player","pfr_id","season","tm","pos"))

snaps = rbind(data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2018.csv")),
              data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2019.csv")),
              data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2020.csv")),
              data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2021.csv")),
              data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2022.csv")),
              data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_2023.csv")))%>%
  select(season,week,team,opponent,position,player,pfr_player_id,offense_snaps,offense_pct)%>%
  rename(pfr_id = "pfr_player_id")%>%
  mutate(played = case_when(offense_snaps>0~TRUE,
                            TRUE~FALSE))

snaps_stats = merge(snaps,adv_offense,by=c("pfr_id","season")) %>%
  group_by(season,week,team)%>%
  filter(position=="WR")%>%
  mutate(snap_rank = round(rank(-offense_pct,ties.method="random")))%>%
  filter(snap_rank<=3)%>%
  mutate(snap_rank = paste0("OPT",snap_rank))

weekly_pass = merge(snaps %>% select(season,week,team,player,offense_pct),
                      weekly_pass,
                      by.x=c("season","week","team","player"),
                      by.y=c("season","week","team","player_display_name"))%>%
  rename("qb_offense_pct" = offense_pct)


snaps_stats_melt = melt(snaps_stats, id.var = c("season","week","team","opponent","snap_rank"))#%>%filter(variable %in% c(...))

snaps_cast = dcast(snaps_stats_melt,season+week+team+opponent~snap_rank+variable,value.var="value") %>%
  mutate(across(.cols = c(8,9,13:22,26,27,31:40,44,45,49:ncol(.)), .fns = as.numeric)) 

qb_weekly_and_season = merge(weekly_pass,season_pass,
                             by.x=c("player_id","player","season","team"),
                             by.y=c("QB_SZN_player_id","QB_SZN_player_display_name","QB_SZN_season","QB_SZN_team"))
qb_opt_df = merge(qb_weekly_and_season,snaps_cast,by=c("season","week","team"))

# now need to bring in defense
defense = data.frame(fread("https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_season_def.csv"))%>%
  dplyr::group_by(season,tm)%>%
  dplyr::summarize(interceptions = sum(int)/ max(g),
                   targets = sum(tgt)/ max(g),
                   completions = sum(cmp)/ max(g),
                   cmp_percent = (sum(cmp)/sum(tgt)),
                   yards = sum(yds)/ max(g),
                   yards_per_cmp = sum(yds) / sum(cmp),
                   yards_per_tgt = sum(yds) / sum(tgt),
                   td = sum(td)/ max(g),
                   rating = sum(rat*gs,na.rm=TRUE) / sum(gs),
                   dadot = sum(dadot * tgt,na.rm=TRUE) / sum(tgt),
                   air_yards = sum(air)/ max(g),
                   yards_after_catch = sum(yac) / max(g),
                   blitz = sum(bltz)/ max(g),
                   hurry = sum(hrry)/ max(g),
                   qbkd = sum(qbkd)/ max(g),
                   sacks = sum(sk)/ max(g),
                   pressures = sum(prss) / max(g),#sum hurry, kd, sacks
                   combined_tackles = sum(comb)/ max(g),
                   missed_tackles = sum(m_tkl) / max(g),
                   missed_tackle_percent = sum(m_tkl) / sum(comb)) %>% 
  filter(season >= 2018, !tm %in% c("2TM","3TM"))%>%
  rename(opponent="tm")
colnames(defense) = paste0("DEF_",colnames(defense))

off_def_merged = merge(qb_opt_df,defense,by.x=c("season","opponent"),by.y=c("DEF_season","DEF_opponent"))
off_def_merged = off_def_merged %>% filter(qb_offense_pct > .98)
#write.csv(off_def_merged,"/Users/jackwerner/Documents/qb_top3WRs_and_opp_defense_2018_2023.csv")
# fix selection
off_def_df_train = off_def_merged %>% select(7,
                                             passing_yards,#qb_offense_pct
                                             12:39,#qb season stats
                                             OPT1_offense_pct,
                                             48:57,
                                             OPT2_offense_pct,
                                             66:75,
                                             OPT3_offense_pct,
                                             84:93,
                                             94:ncol(.)) %>%
  filter(!is.nan(QB_SZN_avg_dakota))

off_def_df_train = off_def_df_train %>% 
  mutate_if(is.character,as.numeric)

# 
# graph_data = off_def_merged %>% 
#   filter(player_display_name == "Josh Allen",
#          season == 2023)
# 
# View(cor(off_def_merged %>% select(passing_yards,63:82)))
# corrplot::corrplot(cor(off_def_merged %>% select(passing_yards,63:82)))
# 
# ggplot(data = graph_data, aes(x=week,y=passing_yards))+
#   geom_hline(yintercept = mean(graph_data$passing_yards),color="red")+
#   geom_line()+
#   geom_line(aes(x=week,y=air_yards+yards_after_catch),color="blue",alpha=.5)+
#   geom_text(aes(y=air_yards+yards_after_catch,label=opponent))+
#   ylim(0,max(graph_data$passing_yards))+
#   theme_minimal()
# 
# ggplot(data = graph_data, aes(x=week,y=completions.x))+
#   geom_hline(yintercept = mean(graph_data$completions.x),color="red")+
#   geom_line()+
#   geom_line(aes(x=week,y=completions.y),color="blue",alpha=.5)+
#   geom_text(aes(y=completions.y,label=opponent))+
#   ylim(0,max(graph_data$completions.x))+
#   theme_minimal()


# real modeling
qb_prop_grid <-  expand.grid(
  nrounds = 500,
  eta = c(.05, .02,.015,.01),#learning rate
  max_depth = c(1,2,3),
  gamma=0,
  colsample_bytree = c(.6,.7,.8),
  min_child_weight = c(.5,.75,1),
  subsample = c(.5,.75,1)  
)

off_def_df_train = off_def_df_train[complete.cases(off_def_df_train),]
qb_prob_model <- train(
  (passing_yards) ~ .,
  data = off_def_df_train,
  method = "xgbTree",
  verbose = TRUE,
  preProc = c("center","scale","corr"),
  tuneGrid = qb_prop_grid,
  trControl = trainControl(method = "cv",  number = 5)) 
plot(qb_prob_model)
qb_prob_model$results[qb_prob_model$results$RMSE == min(qb_prob_model$results$RMSE),]
plot(varImp(qb_prob_model,scale=FALSE))

off_def_df_train$pred_passing_yards = predict(qb_prob_model)
off_def_df_train$error = off_def_df_train$pred_passing_yards - off_def_df_train$passing_yards
hist(off_def_df_train$error)

View(off_def_df_train %>%
  mutate(group = case_when(error >= 45 ~ "high",
                           error < 45 ~ "low")) %>%
  group_by(group) %>%
  summarize_all(mean))






# need to update below columns
off_def_df_train = off_def_merged %>% select(7,
                                             completions,#qb_offense_pct
                                             12:39,#qb season stats
                                             OPT1_offense_pct,
                                             48:57,
                                             OPT2_offense_pct,
                                             66:75,
                                             OPT3_offense_pct,
                                             84:93,
                                             94:ncol(.)) %>%
  filter(!is.nan(QB_SZN_avg_dakota))

off_def_df_train = off_def_df_train %>% 
  mutate_if(is.character,as.numeric)
off_def_df_train = off_def_df_train[complete.cases(off_def_df_train),]

qb_comp_model <- train(
  (completions) ~ .,
  data = off_def_df_train,
  method = "xgbTree",
  verbose = TRUE,
  preProc = c("center","scale","corr"),
  tuneGrid = qb_prop_grid,
  trControl = trainControl(method = "cv",  number = 5)) 
plot(qb_comp_model)
qb_comp_model$results[qb_comp_model$results$RMSE == min(qb_comp_model$results$RMSE),]
plot(varImp(qb_comp_model,scale=FALSE))


# qb szn avg stats
upcoming_qb = season_pass %>% filter(QB_SZN_season == 2023,QB_SZN_player_display_name == "Patrick Mahomes")
# top 3 options - handle NA = 0
upcoming_off = snaps_cast %>% filter(season==2023, team == "KC") %>% tail(1)
# defense
upcoming_def = defense %>% filter(DEF_season == 2023, DEF_opponent == "SF")
upcoming = data.frame(upcoming_qb,upcoming_off %>% mutate_if(is.character,as.numeric),upcoming_def) %>%
  mutate(qb_offense_pct = 1) #%>% mutate_if(is.character,as.numeric)
upcoming  =  upcoming[,colSums(is.na(upcoming))<nrow(upcoming)] 

predict(qb_prob_model,upcoming)




# qb szn avg stats
upcoming_qb_2 = season_pass %>% filter(QB_SZN_season == 2023,QB_SZN_player_display_name == "Brock Purdy")
# top 3 options - handle NA = 0
upcoming_off_2 = snaps_cast %>% filter(season==2023, team == "SF") %>% tail(1)
# defense
upcoming_def_2 = defense %>% filter(DEF_season == 2023, DEF_opponent == "KC")
upcoming_2 = data.frame(upcoming_qb_2,upcoming_off_2 %>% mutate_if(is.character,as.numeric),upcoming_def_2) %>%
  mutate(qb_offense_pct = 1) #%>% mutate_if(is.character,as.numeric)
upcoming_2  =  upcoming_2[,colSums(is.na(upcoming_2))<nrow(upcoming_2)] 

predict(qb_prob_model,upcoming_2)

# probabilities
book_passing_yds = 259.5
pnorm(abs(predict(qb_prob_model,upcoming) - book_passing_yds) / min(qb_prob_model$results$RMSE))

book_passing_yds_2 = 245.5
pnorm(abs(predict(qb_prob_model,upcoming_2) - book_passing_yds_2) / min(qb_prob_model$results$RMSE))







