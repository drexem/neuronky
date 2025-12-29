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

## Implementation Details

### Player History Tracking
Each player has a dictionary with lists for:
- `ace`, `df`, `svpt`, `1stIn`, `1stWon`, `2ndWon`, `SvGms`, `bpSaved`, `bpFaced`
- `results` - 1 for win, 0 for loss

### Rolling Average Calculation
```python
number_of_matches = 5

# Win percentage
win_pct = sum(player_results[-5:]) / 5

# Serve statistics
w_ace_avg = mean(player_aces[-5:])
```

### Data Filtering
- Only matches where **both** players have at least 5 prior matches are kept
- This ensures all rolling statistics are based on complete 5-match windows
- Removed columns: Original match stats (w_ace, l_ace, minutes, etc.)
- Kept columns: Base features + rolling averages only

## Usage for ML Model

### Input Features (X)
**Match Context:**
- `surface`, `tourney_level`, `best_of`, `round`, `tourney_date`, `match_num`

**Player Info:**
- `winner_id`, `winner_seed`, `winner_hand`, `winner_ht`, `winner_ioc`, `winner_age`, `winner_rank`, `winner_rank_points`
- `loser_id`, `loser_seed`, `loser_hand`, `loser_ht`, `loser_ioc`, `loser_age`, `loser_rank`, `loser_rank_points`

**Rolling Statistics (Past Performance):**
- `win_pct_avg_5`, `loser_win_pct_avg_5`
- `w_ace_avg`, `w_df_avg`, `w_svpt_avg`, `w_1stIn_avg`, `w_1stWon_avg`, `w_2ndWon_avg`, `w_SvGms_avg`, `w_bpSaved_avg`, `w_bpFaced_avg`
- `l_ace_avg`, `l_df_avg`, `l_svpt_avg`, `l_1stIn_avg`, `l_1stWon_avg`, `l_2ndWon_avg`, `l_SvGms_avg`, `l_bpSaved_avg`, `l_bpFaced_avg`

### Target Variable (y)
- **Binary classification**: Winner = 1, Loser = 0
- Or create a new column: `winner_won = 1` for all rows

### Data Split Considerations
- ✅ **Temporal split REQUIRED**: Train on earlier dates, test on later dates
- ✅ Sort by `tourney_date` is already done
- ❌ **DO NOT use random split** - would cause data leakage
- Rolling stats are calculated chronologically - respects time dependency

### Example Split
```python
# Split by date
train_df = final_df[final_df['tourney_date'] < 20220101]  # Before 2022
test_df = final_df[final_df['tourney_date'] >= 20220101]   # 2022 onwards
```

## Missing Values
- ❌ No missing values in final dataset
- All matches without 5 prior matches for both players have been filtered out
- Dataset contains only matches with complete rolling statistics

## Output Files

1. `atp_matches_2000_2024.csv` - Raw merged data
2. `atp_matches_2000_2024_with_win_pct.csv` - With rolling stats (includes None values)
3. `atp_matches_2000_2024_filtered.csv` - Filtered (no None values)
4. `atp_matches_2000_2024_final.csv` - **Final dataset** (filtered + original stats removed)

## Future Enhancements
- [ ] Surface-specific rolling stats (e.g., clay court vs hard court performance)
- [ ] Head-to-head statistics between players
- [ ] Expanding window options (last 10, 20 matches)
- [ ] Weighted averages (recent matches weighted more heavily)
- [ ] Momentum features (current winning/losing streak)
- [ ] Tournament-specific stats (performance at specific venues)
- [ ] Form trend (improving vs declining performance)


