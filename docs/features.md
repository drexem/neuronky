# Features Documentation

## Base Features (Available before match)

**Match Context Features:**
- `surface` - Court surface (Hard, Clay, Grass, Carpet)
- `tourney_level` - Tournament category (Grand Slam, Masters, ATP 250/500, etc.)
- `best_of` - Best of 3 or 5 sets
- `round` - Tournament round (R128, R64, R32, R16, QF, SF, F)
- `tourney_date` - Date of the tournament
- `match_num` - Match number within tournament

**Per Player Features:**
- `{winner/loser}_id` - Unique player identifier
- `{winner/loser}_seed` - Tournament seeding position
- `{winner/loser}_name` - Player name
- `{winner/loser}_hand` - Playing hand (R/L)
- `{winner/loser}_ht` - Height in cm
- `{winner/loser}_ioc` - Country code
- `{winner/loser}_age` - Player age at time of match
- `{winner/loser}_rank` - ATP ranking
- `{winner/loser}_rank_points` - ATP ranking points

## Engineered Features (Rolling Statistics from Last 5 Matches)

**Important**: All rolling statistics exclude the current match. Statistics are calculated from each player's last 5 matches before the current match.

### Win Rate
- `win_pct_avg_5` - Winner's win percentage from last 5 matches
  - Range: 0.0 to 1.0
  - 1.0 = won all last 5 matches
  - 0.0 = lost all last 5 matches
  
- `loser_win_pct_avg_5` - Loser's win percentage from last 5 matches
  - Range: 0.0 to 1.0
  - Same calculation as winner

### Serve Statistics (Winner)
- `w_ace_avg` - Average aces per match from last 5 matches
  - Higher = stronger serve
  
- `w_df_avg` - Average double faults per match from last 5 matches
  - Lower = more consistent serve
  
- `w_svpt_avg` - Average serve points per match from last 5 matches
  - Total points played on serve
  
- `w_1stIn_avg` - Average 1st serves in from last 5 matches
  - Higher = better serve accuracy
  
- `w_1stWon_avg` - Average 1st serve points won from last 5 matches
  - Higher = more effective 1st serve
  
- `w_2ndWon_avg` - Average 2nd serve points won from last 5 matches
  - Higher = better 2nd serve effectiveness
  
- `w_SvGms_avg` - Average service games played from last 5 matches
  - Context for other serve stats
  
- `w_bpSaved_avg` - Average break points saved from last 5 matches
  - Higher = better under pressure
  
- `w_bpFaced_avg` - Average break points faced from last 5 matches
  - Lower = more dominant on serve

### Serve Statistics (Loser)
- `l_ace_avg` - Average aces per match from last 5 matches
- `l_df_avg` - Average double faults per match from last 5 matches
- `l_svpt_avg` - Average serve points per match from last 5 matches
- `l_1stIn_avg` - Average 1st serves in from last 5 matches
- `l_1stWon_avg` - Average 1st serve points won from last 5 matches
- `l_2ndWon_avg` - Average 2nd serve points won from last 5 matches
- `l_SvGms_avg` - Average service games played from last 5 matches
- `l_bpSaved_avg` - Average break points saved from last 5 matches
- `l_bpFaced_avg` - Average break points faced from last 5 matches

## Data Processing Pipeline

1. **Merge**: Combine all yearly CSV files (2000-2024) into one dataset
2. **Sort**: Order by `tourney_date` and `match_num` chronologically
3. **Initialize**: Create empty columns for rolling statistics
4. **Track Players**: Build a dictionary tracking each player's match history
   - For each player: store ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced, results
5. **Calculate Rolling Averages**: 
   - For each match, check if both players have at least 5 prior matches
   - If yes, calculate averages from last 5 matches and update the row
   - Then add current match to player history
6. **Filter**: Remove matches where players don't have 5 prior matches (win_pct_avg_5 is None)
7. **Clean**: Drop original match statistics columns (w_ace, l_ace, etc.) - keep only averages
8. **Save**: Export to `atp_matches_2000_2024_final.csv`


