import pandas as pd
import ta

def get_select_indicator_values(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    opens = df[f"{symbol}_open"]
    highs = df[f"{symbol}_high"]
    lows = df[f"{symbol}_low"]
    closes = df[f"{symbol}_close"]
    volumes = df[f"{symbol}_volume"]
        

    adx = ta.trend.ADXIndicator(highs, lows, closes)
    df[f"{symbol}_adx"] = adx.adx()

    df[f"{symbol}_roc"] = ta.momentum.roc(closes)

    bb = ta.volatility.BollingerBands(closes)
    df[f"{symbol}_bbh"] = bb.bollinger_hband()
    
    obv = ta.volume.OnBalanceVolumeIndicator(closes, volumes)
    df[f"{symbol}_obv"] = obv.on_balance_volume()


    kst = ta.trend.KSTIndicator(closes)
    df[f"{symbol}_sig"] = kst.kst_sig()

    mi = ta.trend.MassIndex(highs, lows)
    df[f"{symbol}_mass_index"] = mi.mass_index()

    stc = ta.trend.STCIndicator(closes)
    df[f"{symbol}_stc"] = stc.stc()


    em = ta.volume.EaseOfMovementIndicator(highs, lows, volumes)
    df[f"{symbol}_em"] = em.ease_of_movement()

    cmf = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, volumes)
    df[f"{symbol}_cmf"] = cmf.chaikin_money_flow()


    ppo = ta.momentum.PercentagePriceOscillator(closes)
    df[f"{symbol}_ppo_hist"] = ppo.ppo_hist()
    df[f"{symbol}_ppo_signal"] = ppo.ppo_signal()

    dlr = ta.others.DailyLogReturnIndicator(closes)
    df[f"{symbol}_dlr"] = dlr.daily_log_return()

    dpo = ta.trend.DPOIndicator(closes)
    df[f"{symbol}_dpo"] = dpo.dpo()

    nvi = ta.volume.NegativeVolumeIndexIndicator(closes, volumes)
    df[f"{symbol}_nvi"] = nvi.negative_volume_index()
    
    vpt = ta.volume.VolumePriceTrendIndicator(closes, volumes)
    df[f"{symbol}_vpt"] = vpt.volume_price_trend()

    df.dropna(inplace=True)

    return df