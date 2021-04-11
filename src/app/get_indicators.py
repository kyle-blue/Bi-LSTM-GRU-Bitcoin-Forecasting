import pandas as pd

def get_select_indicator_values(df: pd.DataFrame) -> pd.DataFrame:
    opens = df[f"{SYMBOL_TO_PREDICT}_open"]
    highs = df[f"{SYMBOL_TO_PREDICT}_high"]
    lows = df[f"{SYMBOL_TO_PREDICT}_low"]
    closes = df[f"{SYMBOL_TO_PREDICT}_close"]
    volumes = df[f"{SYMBOL_TO_PREDICT}_volume"]
        

    adx = ta.trend.ADXIndicator(highs, lows, closes)
    df[f"{SYMBOL_TO_PREDICT}_adx"] = adx.adx()

    df[f"{SYMBOL_TO_PREDICT}_roc"] = ta.momentum.roc(closes)

    bb = ta.volatility.BollingerBands(closes)
    df[f"{SYMBOL_TO_PREDICT}_bbh"] = bb.bollinger_hband()
    
    obv = ta.volume.OnBalanceVolumeIndicator(closes, volumes)
    df[f"{SYMBOL_TO_PREDICT}_obv"] = obv.on_balance_volume()


    kst = ta.trend.KSTIndicator(closes)
    df[f"{SYMBOL_TO_PREDICT}_sig"] = kst.kst_sig()

    mi = ta.trend.MassIndex(highs, lows)
    df[f"{SYMBOL_TO_PREDICT}_mass_index"] = mi.mass_index()

    stc = ta.trend.STCIndicator(closes)
    df[f"{SYMBOL_TO_PREDICT}_stc"] = stc.stc()


    em = ta.volume.EaseOfMovementIndicator(highs, lows, volumes)
    df[f"{SYMBOL_TO_PREDICT}_em"] = em.ease_of_movement()

    cmf = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, volumes)
    df[f"{SYMBOL_TO_PREDICT}_cmf"] = cmf.chaikin_money_flow()


    ppo = ta.momentum.PercentagePriceOscillator(closes)
    df[f"{SYMBOL_TO_PREDICT}_ppo_hist"] = ppo.ppo_hist()
    df[f"{SYMBOL_TO_PREDICT}_ppo_signal"] = ppo.ppo_signal()

    dlr = ta.others.DailyLogReturnIndicator(closes)
    df[f"{SYMBOL_TO_PREDICT}_dlr"] = dlr.daily_log_return()

    dpo = ta.trend.DPOIndicator(closes)
    df[f"{SYMBOL_TO_PREDICT}_dpo"] = dpo.dpo()

    nvi = ta.volume.NegativeVolumeIndexIndicator(closes, volumes)
    df[f"{SYMBOL_TO_PREDICT}_nvi"] = nvi.negative_volume_index()
    
    vpt = ta.volume.VolumePriceTrendIndicator(closes, volumes)
    df[f"{SYMBOL_TO_PREDICT}_vpt"] = vpt.volume_price_trend()

    df.dropna(inplace=True)

    return df