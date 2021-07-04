import os
import argparse

import numpy as np
import pandas as pd

import ta

from finrl.marketdata.yahoodownloader import YahooDownloader
import tensortrade.env.default as default
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, AAPL, LTC, Instrument
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.env.generic.components.renderer import AggregateRenderer

from tensortrade.agents import DQNAgent, A2CAgent

tickers = ['AAPL', 'GOOG'] 

start_date = '2009-01-01'
end_date = '2021-07-01'

def prepare_data():
  exchange = Exchange('exchange', service=execute_order, 
    options=ExchangeOptions(commission=0.01))
  portfolio = Portfolio(USD, [
      Wallet(exchange, 100000 * USD),
  ])
  
  ex_features = []
  renderer_feed = None
  for ticker in tickers:
    renderer_feed = prepare_ticker(exchange, ticker, ex_features, portfolio)

  return portfolio, DataFeed(ex_features), renderer_feed

def download(enable_cache, start_date, end_date, ticker):
  path = f'data/df_{start_date}_{end_date}_{ticker}.pkl'
  if enable_cache:
    if os.path.exists(path):
      print('loading from cache')
      return pd.read_pickle(path)
  downloader = YahooDownloader(start_date, end_date, [ticker,])
  data = downloader.fetch_data()
  if enable_cache:
    print('saving to cache')
    data.to_pickle(path)  
  return data

def prepare_ticker(exchange, ticker, ex_features, portfolio):
  data = download(True, start_date, end_date, ticker)  

  ticker_ins = Instrument(ticker, 2, f'{ticker} stock')
  portfolio.add(Wallet(exchange, 0 * ticker_ins))

  ex_data = pd.concat([
      data.add_prefix(f'{ticker}:'),
  ], axis=1)

  exchange(
      Stream.source(list(ex_data[f'{ticker}:close']), dtype="float").rename(f'USD-{ticker}'),
  )

  # Add all features for bitstamp aapl & goog
  ex_ticker = ex_data.loc[:, [name.startswith(ticker) and 'date' not in name for name in ex_data.columns]]

  ta.add_all_ta_features(
      ex_ticker,
      colprefix=f'{ticker}:',
      **{k: f'{ticker}:' + k for k in ['open', 'high', 'low', 'close', 'volume']}
  )

  ex_streams = [
      Stream.source(list(ex_ticker[c]), dtype="float").rename(c) for c in ex_ticker.columns
  ]

  def select_features(prefix, names, feature_streams):
    for name in names:
      full_name = f'{prefix}:{name}'
      feature_streams[full_name] = Stream.select(ex_streams, lambda s: s.name == full_name)

  def add_features(prefix, feature_streams):
    return [
        # feature_streams[f'{prefix}:open'],
        # feature_streams[f'{prefix}:close'],
        # (feature_streams[f'{prefix}:open'] - feature_streams[f'{prefix}:close'].lag()).rename(f'{prefix}:o_c'),
        feature_streams[f'{prefix}:close'].log().diff().rename(f'{prefix}:log_c_diff'),
        feature_streams[f'{prefix}:volume'].log().diff().rename(f'{prefix}:log_v_diff'),
        feature_streams[f'{prefix}:trend_macd_diff'],
        feature_streams[f'{prefix}:momentum_rsi'],
        # feature_streams[f'{prefix}:day'],
    ]

  feature_streams = {}
  feature_names = ['open', 'close', 'low', 'high', 'volume', 'trend_macd_diff', 'momentum_rsi']
  select_features(ticker, feature_names, feature_streams)

  ex_features.extend(add_features(ticker, feature_streams))

  return DataFeed([                 
      Stream.source(list(data["date"])).rename("date"),
      Stream.source(list(data["open"]), dtype="float").rename("open"),
      Stream.source(list(data["close"]), dtype="float").rename("close"),
      Stream.source(list(data["high"]), dtype="float").rename("high"),
      Stream.source(list(data["low"]), dtype="float").rename("low"),
      Stream.source(list(data["volume"]), dtype="float").rename("volume"),
  ])


def main(args):
  # aapl_downloader = YahooDownloader(start_date, end_date, ['AAPL',])
  # goog_downloader = YahooDownloader(start_date, end_date, ['GOOG',])

  # aapl_data = aapl_downloader.fetch_data()
  # goog_data = goog_downloader.fetch_data()

  # AAPL = Instrument('AAPL', 2, 'Apple stock')
  # GOOG = Instrument('GOOG', 2, 'Google stock')

  # # cdd = CryptoDataDownload()

  # # aapl_data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
  # # goog_data = cdd.fetch("Bitfinex", "USD", "ETH", "1h")

  # # AAPL = Instrument('AAPL', 8, 'Apple stock')
  # # GOOG = Instrument('GOOG', 8, 'Google stock')


  # ex_data = pd.concat([
  #     aapl_data.add_prefix("AAPL:"),
  #     goog_data.add_prefix("GOOG:"),
  # ], axis=1)

  # ex = Exchange("ex", service=execute_order, options=ExchangeOptions(commission=0.01))(
  #     Stream.source(list(ex_data['AAPL:close']), dtype="float").rename("USD-AAPL"),
  #     Stream.source(list(ex_data['GOOG:close']), dtype="float").rename("USD-GOOG")
  # )

  # # Add all features for bitstamp aapl & goog
  # ex_aapl = ex_data.loc[:, [name.startswith("AAPL") and 'date' not in name for name in ex_data.columns]]
  # ex_goog = ex_data.loc[:, [name.startswith("GOOG") and 'date' not in name for name in ex_data.columns]]

  # ta.add_all_ta_features(
  #     ex_aapl,
  #     colprefix="AAPL:",
  #     **{k: "AAPL:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
  # )

  # ta.add_all_ta_features(
  #     ex_goog,
  #     colprefix="GOOG:",
  #     **{k: "GOOG:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
  # )

  # ex_streams = [
  #     Stream.source(list(ex_aapl[c]), dtype="float").rename(c) for c in ex_aapl.columns
  # ]
  # ex_streams += [
  #     Stream.source(list(ex_goog[c]), dtype="float").rename(c) for c in ex_goog.columns
  # ]

  # def eq(a, b):
  #   # print(a, b)
  #   return a == b

  # def select_features(prefix, names, feature_streams):
  #   for name in names:
  #     full_name = f'{prefix}:{name}'
  #     feature_streams[full_name] = Stream.select(ex_streams, lambda s: eq(s.name, full_name))

  # def add_features(prefix, feature_streams):
  #   return [
  #       # feature_streams[f'{prefix}:open'],
  #       # feature_streams[f'{prefix}:close'],
  #       # (feature_streams[f'{prefix}:open'] - feature_streams[f'{prefix}:close'].lag()).rename(f'{prefix}:o_c'),
  #       feature_streams[f'{prefix}:close'].log().diff().rename(f'{prefix}:log_c_diff'),
  #       feature_streams[f'{prefix}:volume'].log().diff().rename(f"{prefix}:log_v_diff"),
  #       feature_streams[f'{prefix}:trend_macd_diff'],
  #       feature_streams[f'{prefix}:momentum_rsi'],
  #       # feature_streams[f'{prefix}:day'],
  #   ]

  # feature_streams = {}
  # feature_names = ['open', 'close', 'low', 'high', 'volume', 'trend_macd_diff', 'momentum_rsi']
  # select_features('AAPL', feature_names, feature_streams)
  # select_features('GOOG', feature_names, feature_streams)

  # ex_features = []
  # ex_features += add_features('AAPL', feature_streams)
  # ex_features += add_features('GOOG', feature_streams)

  # feed = DataFeed(ex_features)

  # portfolio = Portfolio(USD, [
  #     Wallet(ex, 100000 * USD),
  #     Wallet(ex, 0 * GOOG),
  #     Wallet(ex, 0 * AAPL),
  # ])

  portfolio, feed, renderer_feed = prepare_data()

  # renderer_feed_goog = DataFeed([                 
  #     Stream.source(list(goog_data["date"])).rename("date"),
  #     Stream.source(list(goog_data["open"]), dtype="float").rename("open"),
  #     Stream.source(list(goog_data["close"]), dtype="float").rename("close"),
  #     Stream.source(list(goog_data["high"]), dtype="float").rename("high"),
  #     Stream.source(list(goog_data["low"]), dtype="float").rename("low"),
  #     Stream.source(list(goog_data["volume"]), dtype="float").rename("volume"),
  # ])

  env = default.create(
      portfolio=portfolio,
      action_scheme="managed-risk",
      reward_scheme="risk-adjusted",
      feed=feed,
      renderer_feed=renderer_feed,
      renderer=default.renderers.MatplotlibTradingChart(),
      window_size=64,
      min_periods=32,
  )

  agent = DQNAgent(env, policy_network_path=args.model_path)
  # agent = A2CAgent(env)

  # agent.train(n_steps=27473, n_episodes=20, save_path="agents/", render_interval=200, memory_capacity=4096, eps_decay_steps=27473*15, update_target_every=200)
  # for stock

  agent.train(n_steps=3000, n_episodes=200, save_path=f'agents/{args.job_id}', save_every=1,
      render_interval=200, memory_capacity=4096, eps_decay_steps=3000*150, 
      update_target_every=200)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model_path')
  parser.add_argument('-b', '--job_id', default='default')
  args = parser.parse_args() 
  main(args)