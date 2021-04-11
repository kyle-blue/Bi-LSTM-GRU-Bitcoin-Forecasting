import pandas as pd
import ta

def get_select_indicator_values(df: pd.DataFrame) -> pd.DataFrame:
    opens = df["open"]
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    volumes = df["volume"]
        

    adx = ta.trend.ADXIndicator(highs, lows, closes)
    df["adx"] = adx.adx()

    df["roc"] = ta.momentum.roc(closes)

    bb = ta.volatility.BollingerBands(closes)
    df["bbh"] = bb.bollinger_hband()
    
    obv = ta.volume.OnBalanceVolumeIndicator(closes, volumes)
    df["obv"] = obv.on_balance_volume()


    kst = ta.trend.KSTIndicator(closes)
    df["sig"] = kst.kst_sig()

    mi = ta.trend.MassIndex(highs, lows)
    df["mass_index"] = mi.mass_index()

    stc = ta.trend.STCIndicator(closes)
    df["stc"] = stc.stc()


    em = ta.volume.EaseOfMovementIndicator(highs, lows, volumes)
    df["em"] = em.ease_of_movement()

    cmf = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, volumes)
    df["cmf"] = cmf.chaikin_money_flow()


    ppo = ta.momentum.PercentagePriceOscillator(closes)
    df["ppo_hist"] = ppo.ppo_hist()
    df["ppo_signal"] = ppo.ppo_signal()

    dlr = ta.others.DailyLogReturnIndicator(closes)
    df["dlr"] = dlr.daily_log_return()

    dpo = ta.trend.DPOIndicator(closes)
    df["dpo"] = dpo.dpo()

    nvi = ta.volume.NegativeVolumeIndexIndicator(closes, volumes)
    df["nvi"] = nvi.negative_volume_index()
    
    vpt = ta.volume.VolumePriceTrendIndicator(closes, volumes)
    df["vpt"] = vpt.volume_price_trend()

    df.dropna(inplace=True)

    return df