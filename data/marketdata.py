import requests
import pandas as pd
import io
import time
import os
from datetime import date
from selenium import webdriver
from splinter import Browser

class MarketData:
    def __init__(self, data_dir='data/ticker_info'):
        self.data_dir = data_dir
        self.nse_file = os.path.join(self.data_dir, 'nse_data.csv')
        self.bse_file = os.path.join(self.data_dir, 'bse_data.csv')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.nse_list = []
        self.nse_isin_mapping = {}
        self.nse_mapping = {}

        self.bse_list = []
        self.nse_isin_mapping = {}
        self.nse_mapping = {}
        
        self.load_nse_data()
        self.load_bse_data()
        self.combine_list()
    
    def load_nse_data(self):
        if os.path.exists(self.nse_file):
            self.nse = pd.read_csv(self.nse_file)
            self.nse_list = (self.nse['SYMBOL']+'.NS').tolist()
            self.nse_mapping = dict(zip(self.nse['NAME OF COMPANY']+ ' (NSE)' , self.nse['SYMBOL']+'.NS'))
            self.nse_isin_mapping = dict(zip(self.nse['NAME OF COMPANY'] + ' (NSE)', self.nse[' ISIN NUMBER']))
        else:
            self.update_nse_list()
    
    def load_bse_data(self):
        if os.path.exists(self.bse_file):
            self.bse = pd.read_csv(self.bse_file)
            self.bse_list = (self.bse['Security Id']+'.BO').tolist()
            self.bse_mapping = dict(zip(self.bse['Security Name'] + ' (BSE)', self.bse['Security Id']+'.BO'))
            self.bse_isin_mapping = dict(zip(self.bse['Security Name'] + ' (BSE)', self.bse['ISIN No']))
        else:
            self.update_bse_list()
    
    def update_nse_list(self):
        nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        with requests.Session() as s:
            s.headers.update(headers)
            r = s.get(nse_url)
        
        response_text = r.text.strip()
        if response_text.startswith("<!DOCTYPE HTML") and "503 Service Unavailable" in response_text:
            print("NSE data is unavailable. Skipping update.")
            return
        
        df_nse = pd.read_csv(io.BytesIO(r.content))
        df_nse.to_csv(self.nse_file, index=False)
        self.load_nse_data()
    
    def update_bse_list(self):
        bse_link = "https://bseindia.com/corporates/List_Scrips.html"
        
        options = webdriver.ChromeOptions()
        prefs = {"download.default_directory": self.data_dir}
        options.add_experimental_option("prefs", prefs)
        
        browser = Browser('chrome', options=options, headless=True)
        browser.visit(bse_link)
        
        browser.find_by_id('ddlsegment').select("Equity")
        browser.find_by_id('ddlstatus').select("Active")
        browser.find_by_id('btnSubmit').click()
        
        browser.is_element_present_by_text("Issuer Name")
        time.sleep(5)
        
        browser.find_by_id('lnkDownload').click()
        time.sleep(5)
        
        downloaded_file = os.path.join(self.data_dir, "Equity.csv")
        if os.path.exists(downloaded_file):
            os.rename(downloaded_file, self.bse_file)
        browser.quit()
        self.load_bse_data()

    def combine_list(self):
        self.ticker_list = self.nse_list+self.bse_list
        self.company_list = list(self.nse_mapping.keys()) + list(self.bse_mapping.keys())
        self.company_mapping = self.nse_isin_mapping | self.bse_isin_mapping

    
if __name__ == "__main__":
    
    market = MarketData()
    print(market.nse_list[:5])
    print(market.bse_list[:5])
    print(market.company_list[-5:])