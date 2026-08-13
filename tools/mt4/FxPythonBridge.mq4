#property strict

input string CommandFile = "fx_bridge_commands.csv";
input string ResultFile = "fx_bridge_results.csv";
input string HeartbeatFile = "fx_bridge_heartbeat.csv";
input int Slippage = 30;
input int MagicNumber = 60251364;

string last_command_id = "";

int OnInit()
{
   EventSetTimer(1);
   WriteHeartbeat();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTick()
{
   WriteHeartbeat();
   ProcessCommand();
}

void OnTimer()
{
   WriteHeartbeat();
   ProcessCommand();
}

void WriteHeartbeat()
{
   int handle = FileOpen(HeartbeatFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;
   FileWrite(
      handle,
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
      AccountNumber(),
      AccountBalance(),
      AccountEquity(),
      AccountFreeMargin(),
      IsTradeAllowed()
   );
   FileClose(handle);
   WriteMarketSnapshot(Symbol());
   WritePositionsSnapshot();
   WriteHistorySnapshot();
}

string SafeFileSymbol(string symbol)
{
   string safe = symbol;
   StringReplace(safe, ".", "_");
   StringReplace(safe, "#", "_");
   StringReplace(safe, "/", "_");
   StringReplace(safe, "\\", "_");
   return safe;
}

void WriteMarketSnapshot(string symbol)
{
   RefreshRates();
   string safe = SafeFileSymbol(symbol);
   double bid = MarketInfo(symbol, MODE_BID);
   double ask = MarketInfo(symbol, MODE_ASK);
   int tick_handle = FileOpen("fx_bridge_tick_" + safe + ".csv", FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(tick_handle != INVALID_HANDLE)
   {
      FileWrite(
         tick_handle,
         (int)TimeCurrent(),
         bid,
         ask,
         MarketInfo(symbol, MODE_POINT),
         MarketInfo(symbol, MODE_DIGITS),
         MarketInfo(symbol, MODE_MINLOT),
         MarketInfo(symbol, MODE_MAXLOT),
         MarketInfo(symbol, MODE_LOTSTEP),
         MarketInfo(symbol, MODE_STOPLEVEL),
         MarketInfo(symbol, MODE_FREEZELEVEL),
         MarketInfo(symbol, MODE_LOTSIZE),      // col[10] contract_size
         MarketInfo(symbol, MODE_TICKVALUE)     // col[11] tick_value
      );
      FileClose(tick_handle);
   }
   AppendTickHistory(symbol, safe, bid, ask);

   WriteRates(symbol, PERIOD_D1, "D1", 100);
   WriteRates(symbol, PERIOD_H4, "H4", 100);
   WriteRates(symbol, PERIOD_H1, "H1", 300);
   WriteRates(symbol, PERIOD_M30, "M30", 100);
   WriteRates(symbol, PERIOD_M15, "M15", 150);
   WriteRates(symbol, PERIOD_M5, "M5", 150);
   WriteRates(symbol, PERIOD_M1, "M1", 7500);
}

void AppendTickHistory(string symbol, string safe, double bid, double ask)
{
   static string last_signature = "";
   if(bid <= 0 && ask <= 0)
      return;

   int digits = (int)MarketInfo(symbol, MODE_DIGITS);
   string signature = symbol + "|" + TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + "|" + DoubleToString(bid, digits) + "|" + DoubleToString(ask, digits);
   if(signature == last_signature)
      return;
   last_signature = signature;

   int handle = FileOpen("fx_bridge_ticks_" + safe + ".csv", FILE_READ | FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, (int)TimeCurrent(), bid, ask);
   FileClose(handle);
}

void WriteRates(string symbol, int timeframe, string label, int count)
{
   string path = "fx_bridge_rates_" + SafeFileSymbol(symbol) + "_" + label + ".csv";
   int handle = FileOpen(path, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;

   for(int shift = count; shift >= 1; shift--)
   {
      datetime bar_time = iTime(symbol, timeframe, shift);
      if(bar_time <= 0)
         continue;
         FileWrite(
         handle,
         (int)bar_time,
         iOpen(symbol, timeframe, shift),
         iHigh(symbol, timeframe, shift),
         iLow(symbol, timeframe, shift),
         iClose(symbol, timeframe, shift),
         iVolume(symbol, timeframe, shift)
      );
   }
   FileClose(handle);
}

void WritePositionsSnapshot()
{
   int handle = FileOpen("fx_bridge_positions.csv", FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;

   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderMagicNumber() != MagicNumber)
         continue;
      FileWrite(
         handle,
         OrderTicket(),
         OrderSymbol(),
         OrderLots(),
         OrderType(),
         OrderOpenPrice(),
         OrderStopLoss(),
         OrderTakeProfit(),
         OrderProfit(),
         OrderComment()
      );
   }
   FileClose(handle);
}

void WriteHistorySnapshot()
{
   int handle = FileOpen("fx_bridge_history.csv", FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;

   for(int i = 0; i < OrdersHistoryTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
         continue;
      FileWrite(
         handle,
         OrderTicket(),
         OrderSymbol(),
         OrderLots(),
         OrderType(),
         OrderOpenPrice(),
         OrderClosePrice(),
         OrderStopLoss(),
         OrderTakeProfit(),
         OrderProfit(),
         TimeToString(OrderOpenTime(), TIME_DATE | TIME_SECONDS),
         TimeToString(OrderCloseTime(), TIME_DATE | TIME_SECONDS),
         OrderComment(),
         OrderMagicNumber()
      );
   }
   FileClose(handle);
}

void WriteResult(string command_id, string status, string message, int ticket, int error_code)
{
   // NOTE: capture error_code BEFORE FileOpen — FileOpen resets GetLastError() to 0.
   int handle = FileOpen(ResultFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;
   FileWrite(handle, command_id, status, message, ticket, error_code);
   FileClose(handle);
}

void ProcessCommand()
{
   int handle = FileOpen(CommandFile, FILE_READ | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
      return;

   string command_id = FileReadString(handle);
   string action = FileReadString(handle);
   StringToUpper(action);
   string symbol = FileReadString(handle);
   double lots = FileReadNumber(handle);
   double stop_loss = FileReadNumber(handle);
   double take_profit = FileReadNumber(handle);
   string comment = FileReadString(handle);
   FileClose(handle);

   if(command_id == "" || command_id == last_command_id)
      return;

   last_command_id = command_id;

   if(!IsTradeAllowed())
   {
      WriteResult(command_id, "ERROR", "MT4 reports trading is not allowed for this EA/account. Check Tools > Options > Expert Advisors and broker permissions.", -1, 0);
      return;
   }

   if(action == "BUY" || action == "SELL")
      SendMarketOrder(command_id, action, symbol, lots, stop_loss, take_profit, comment);
   else if(action == "CLOSE")
      CloseSymbolOrders(command_id, symbol, (int)lots);
   else if(action == "MODIFY")
      ModifySymbolOrders(command_id, symbol, stop_loss, take_profit, (int)lots);
   else
      WriteResult(command_id, "ERROR", "Unknown action", -1, 0);
}

void SendMarketOrder(string command_id, string action, string symbol, double lots, double stop_loss, double take_profit, string comment)
{
   RefreshRates();
   int type = (action == "BUY") ? OP_BUY : OP_SELL;
   double price = (type == OP_BUY) ? MarketInfo(symbol, MODE_ASK) : MarketInfo(symbol, MODE_BID);
   int digits = (int)MarketInfo(symbol, MODE_DIGITS);
   double point = MarketInfo(symbol, MODE_POINT);
   price = NormalizeDouble(price, digits);

   // Enforce broker minimum stop distance (MODE_STOPLEVEL points from price)
   // Violating this causes ERR_INVALID_STOPS (retcode 130).
   double min_dist = MarketInfo(symbol, MODE_STOPLEVEL) * point;
   if(stop_loss > 0)
   {
      stop_loss = NormalizeDouble(stop_loss, digits);
      if(type == OP_BUY  && price - stop_loss < min_dist)
         stop_loss = NormalizeDouble(price - min_dist, digits);
      if(type == OP_SELL && stop_loss - price < min_dist)
         stop_loss = NormalizeDouble(price + min_dist, digits);
   }
   if(take_profit > 0)
   {
      take_profit = NormalizeDouble(take_profit, digits);
      if(type == OP_BUY  && take_profit - price < min_dist)
         take_profit = NormalizeDouble(price + min_dist, digits);
      if(type == OP_SELL && price - take_profit < min_dist)
         take_profit = NormalizeDouble(price - min_dist, digits);
   }

   int error_before = GetLastError(); // clear any stale error
   int ticket = OrderSend(symbol, type, lots, price, Slippage, stop_loss, take_profit, comment, MagicNumber, 0, clrDodgerBlue);
   int send_error = GetLastError();
   if(ticket < 0)
      WriteResult(command_id, "ERROR", "OrderSend failed err=" + IntegerToString(send_error), ticket, send_error);
   else
      WriteResult(command_id, "OK", "Order opened", ticket, 0);
}

void CloseSymbolOrders(string command_id, string symbol, int target_ticket)
{
   bool closed_any = false;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderSymbol() != symbol || OrderMagicNumber() != MagicNumber)
         continue;
      if(target_ticket > 0 && OrderTicket() != target_ticket)
         continue;

      RefreshRates();
      int type = OrderType();
      double price = (type == OP_BUY) ? MarketInfo(symbol, MODE_BID) : MarketInfo(symbol, MODE_ASK);
      int digits = (int)MarketInfo(symbol, MODE_DIGITS);
      price = NormalizeDouble(price, digits);

      if(OrderClose(OrderTicket(), OrderLots(), price, Slippage, clrTomato))
         closed_any = true;
      else
      {
         int err = GetLastError();
         WriteResult(command_id, "ERROR", "OrderClose failed err=" + IntegerToString(err), OrderTicket(), err);
         return;
      }
   }

   if(closed_any)
      WriteResult(command_id, "OK", "Orders closed", 0, 0);
   else
      WriteResult(command_id, "OK", "No matching orders to close", 0, 0);
}

void ModifySymbolOrders(string command_id, string symbol, double stop_loss, double take_profit, int target_ticket)
{
   bool modified_any = false;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderSymbol() != symbol || OrderMagicNumber() != MagicNumber)
         continue;
      if(target_ticket > 0 && OrderTicket() != target_ticket)
         continue;

      int digits = (int)MarketInfo(symbol, MODE_DIGITS);
      double point    = MarketInfo(symbol, MODE_POINT);
      double min_dist = MarketInfo(symbol, MODE_STOPLEVEL) * point;
      double open_px  = OrderOpenPrice();
      int    otype    = OrderType();

      double sl = stop_loss > 0 ? NormalizeDouble(stop_loss, digits) : OrderStopLoss();
      double tp = take_profit > 0 ? NormalizeDouble(take_profit, digits) : OrderTakeProfit();

      // Clamp SL/TP to meet broker minimum stop distance for modify as well
      if(sl > 0)
      {
         if(otype == OP_BUY  && open_px - sl < min_dist) sl = NormalizeDouble(open_px - min_dist, digits);
         if(otype == OP_SELL && sl - open_px < min_dist) sl = NormalizeDouble(open_px + min_dist, digits);
      }
      if(tp > 0)
      {
         if(otype == OP_BUY  && tp - open_px < min_dist) tp = NormalizeDouble(open_px + min_dist, digits);
         if(otype == OP_SELL && open_px - tp < min_dist) tp = NormalizeDouble(open_px - min_dist, digits);
      }

      if(!OrderModify(OrderTicket(), open_px, sl, tp, 0, clrDodgerBlue))
      {
         int err = GetLastError();
         WriteResult(command_id, "ERROR", "OrderModify failed err=" + IntegerToString(err), OrderTicket(), err);
         return;
      }
      modified_any = true;
   }

   if(modified_any)
      WriteResult(command_id, "OK", "Orders modified", 0, 0);
   else
      WriteResult(command_id, "OK", "No matching orders to modify", 0, 0);
}
