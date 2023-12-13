# Example sheet for performing the whole operation, 
# Remember to delete the numbered file name and "masterFile" from drive before running this sheet
# sheetCreateEdit.py should be together with this file to import as library

from drive_utils.sheetCreateEdit import spreadsheetCreation,spreadSheetEditor
import os
from datetime import datetime, timedelta

class DriveHandler:
    def __init__(self):
        self.parentName ="LogFld"                                        # Parent folder name
        self.cred_drive = r'drive_utils/credentials/Aouth2.json'         # Credential.json file for aouth 2.0 
        self.cred_spread = r'drive_utils/credentials/serviceAcc.json'    # Credential.json file for service account 
        self.token = r'drive_utils/credentials/token.json'               # token.json file, keep it as it is as it will create itself if its null
        self.filePath = r'log'                                           # file path to the parent folder of the text file
        self.sheet_ID = 6868                                             # ID of the chart sheet in spreasheet, need to predefined        
        
    def post(self):
        # get the masterfile name, to record accumulated count each day
        currentYear  = datetime.now().year
        currentMonth = datetime.now().month
        currentDay   = datetime.now().day
        if currentDay == 1:
            currentYear  = (datetime.now() - timedelta(1)).year 
            currentMonth = (datetime.now() - timedelta(1)).month            
        masterFile = f"masterFile_{currentYear}/{currentMonth}"     # Name of the master sheet to record accumulated count each day
        
        logfiles = sorted(os.listdir(self.filePath))
        for logfile in logfiles:
            fileName = logfile
            fp = os.path.join(self.filePath,logfile)

            sheetCreation = spreadsheetCreation(self.cred_drive,self.token,self.parentName)        # initialize credential for creating
            sheetEditor = spreadSheetEditor(self.cred_spread,str(fileName),fp)           # initialize credential for editing
            
            # if spreadsheet of that day does not exist, then will create a new one and import the data in it
            if sheetCreation.findSheet(str(fileName)) == False : 
                sheetCreation.createSheet(str(fileName))
                sheetEditor.dfImport()
                print(f"created {str(fileName)}")
            else:
                print(f"{str(fileName)} existed, skip to next logfile")
                continue

            # will chart a bar chart to a new sheet, if chart sheet exist it will not create a new chart sheet
            sheetEditor.dfChart(str(fileName),self.sheet_ID,1,"hour",2)

            # if masterfile does not exist, it will create a new spreadsheet of masterfile, and make a new sheet to create chart
            if sheetCreation.findSheet(masterFile) == False:
                print("masterfile does not exist")
                sheetCreation.createSheet(masterFile)
                sheetEditor.dfChart(masterFile,self.sheet_ID,0,"day",1) #<----
            
            # Append the accumulated count to the masterfile
            sheetEditor.dfAppend(masterFile)
            print(f"write to {fileName}")    
