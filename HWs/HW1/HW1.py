# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:50:26 2021

@author: izely
"""
import random
random.seed(77549)

class Portfolio(object) :
    
    def __init__(self):
    
        self.cash = 0
        self.stocks = {}
        self.mutualFunds = {}
        self.transactionHistory = []
    
    def history(self):
        i=0
        print("Transaction History:")
        print("  Name\t\tType\t\tSymbol\t\tAmount\t\tPrice")
        for t in self.transactionHistory:
            i = i+1
            print("{}-{}".format(i,t))
        print("")


    def __str__(self):
        print("Portfolio:")
        print("cash: {} $".format(self.cash))
        print("stocks:")
        for s in self.stocks:
            print("{} , {}".format(self.stocks[s].your_amount, self.stocks[s].symbol))
        print("mutual funds:")
        for m in self.mutualFunds:
            print("{} , {}".format(self.mutualFunds[m].your_amount, self.mutualFunds[m].symbol))
        return ""    

    def addCash(self,cash):
        self.cash = self.cash + cash
        self.transactionHistory.append("CASH\t\tADD\t\t$\t-\t{}".format(cash))
        
    def buyStock(self, amount, stock): 
        self.cash = self.cash - (amount*stock.price)
        stock.your_amount = stock.your_amount + amount
        self.stocks[stock.symbol] = stock
        self.transactionHistory.append("STOCK\t\tBUY\t\t{}\t{}\t{}".format(stock.symbol, amount, stock.price))
 
    def buyMutualFund(self, amount, mutualFund): 
        self.cash = self.cash - amount
        mutualFund.your_amount = mutualFund.your_amount + amount
        self.mutualFunds[mutualFund.symbol] = mutualFund
        self.transactionHistory.append("MUTUAL_FUND\tBUY\t\t{}\t{}\t{}".format(mutualFund.symbol, amount, mutualFund.price))



    def withdrawCash(self,cash):  
        self.cash = self.cash - cash
        self.transactionHistory.append("CASH\t\tWITHDRAW\t$\t-\t{}".format(cash))
        
    def sellStock(self,symbol,amount):    
       if symbol in self.stocks.keys():
            if self.stocks[symbol].your_amount > amount:
                self.stocks[symbol].your_amount = self.stocks[symbol].your_amount -amount
                low_limit= 0.5* self.stocks[symbol].price
                up_limit = 1.5 * self.stocks[symbol].price
                self.cash = self.cash + random.uniform(low_limit,up_limit)
                self.transactionHistory.append("STOCK\t\tSELL\t\t{}\t{}\t{}".format(symbol, amount, self.stocks[symbol].price))
            else:
                print("You don't have enough amount of stock")
       else:
            print("You don't have {} stock in your portfolio", symbol)
    
    def sellMutualFund(self,symbol,amount):   
        if symbol in self.mutualFunds.keys():
            if  self.mutualFunds[symbol].your_amount > amount:
                self.mutualFunds[symbol].your_amount = self.mutualFunds[symbol].your_amount - amount
                self.cash = self.cash + random.uniform(0.9, 1.2)
                self.transactionHistory.append("MUTUAL_FUND\tSELL\t\t{}\t{}\t{}".format(symbol, amount, 1))
            else:
                print("You don't have enough amount of mutual fund")
        else:
            print("You don't have {} mutual fund in your portfolio", symbol)


class Stock(object):  
    
    def __init__(self, price, symbol):
        self.price  = price
        self.symbol = symbol
        self.your_amount = 0 
        
class MutualFund(object):

    def __init__(self, symbol):
        self.symbol = symbol
        self.price = 1
        self.your_amount = 0  
        

def main():
  
    portfolio = Portfolio() #Creates a new portfolio
    portfolio.addCash(300.50) #Adds cash to the portfolio
    s = Stock(20, "HFH") #Create Stock with price 20 and symbol "HFH"
    portfolio.buyStock(5, s) #Buys 5 shares of stock s
    mf1 = MutualFund("BRT") #Create MF with symbol "BRT"
    mf2 = MutualFund("GHT") #Create MF with symbol "GHT"
    portfolio.buyMutualFund(10.3, mf1) #Buys 10.3 shares of "BRT"
    portfolio.buyMutualFund(2, mf2) #Buys 2 shares of "GHT"
    print(portfolio) #Prints portfolio
    #cash: $140.50
    #stock: 5 HFH
    
    #mutual funds: 10.33 BRT
    # 2 GHT
    portfolio.sellMutualFund("BRT", 3) #Sells 3 shares of BRT
    portfolio.sellStock("HFH", 1) #Sells 1 share of HFH
    portfolio.withdrawCash(50) #Removes $50
    portfolio.history() #Prints a list of all transactions
    #ordered by time

    print(portfolio)  # Prints portfolio

main()        
        
    
