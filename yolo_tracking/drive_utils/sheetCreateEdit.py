from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

import os.path

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Create parents, spreadsheet, and find if sheet existed(even from trash)
class spreadsheetCreation:

    # Initialize drive credentail token for drive if token not existed, service used and parent name
    # Only use Aouth credential for this, service account credential should not be used
    def __init__(self, cred_aouth,token,parent_name):
        SCOPES = ["https://www.googleapis.com/auth/drive"]

        creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
        if os.path.exists(token):
            creds = Credentials.from_authorized_user_file(token, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_aouth, SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token, "w") as token:
                token.write(creds.to_json())
        
        self.service = build("drive", "v3", credentials=creds)
        self.parent_name = parent_name

    #create parent folder
    def createParents(self):
        file_metadata = {
            "name" : self.parent_name,
            "mimeType": "application/vnd.google-apps.folder"
        }

        file = self.service.files().create(
            body=file_metadata,
            fields = "id"
            ).execute()
        
        folder_id = file.get('id') 

        return folder_id    

    # find if certain spreadsheet exist inside drive
    def findSheet(self,sheet_name):
        response = self.service.files().list(
            q="name = '"+ sheet_name +"' and mimeType='application/vnd.google-apps.spreadsheet'",
            spaces = 'drive'
        ).execute()

        if not response['files']:
            return False
        else:
            return True

    # create spreadsheet in drive, auto create parent if does not exist
    # Edit permission is also being shared with service account for it to be able edit later
    def createSheet(self,sheet_name):

        response = self.service.files().list(
            q="name = '"+ self.parent_name +"' and mimeType='application/vnd.google-apps.folder'",
            spaces = 'drive'
        ).execute()

        if not response['files']:
            folder_id = self.createParents()
        else:
            folder_id = response['files'][0]['id']

        body = {
            "name": sheet_name,
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "parents": [folder_id]
        }
        
        spreadsheet = self.service.files().create(body=body,
                                    fields="id", 
                                    supportsAllDrives=True).execute()
        
        permission = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': "ms-ang@libcounttest.iam.gserviceaccount.com"
        }

        req = self.service.permissions().create(
            fileId=spreadsheet['id'],
            body=permission,
            fields='id'
        ).execute()

# -Edit spreadsheet
# -mainly import data from txt file to spreadsheet, 
# -create chart sheet(only once for every spredsheet)
# -Append last row data(accumulated count each day) to the master sheet(for counting entire week, month, or year of your choice)
 
class spreadSheetEditor:
    # Authorize spreadsheet credential for editing spreadsheet files using service account
    # initialize sheet name to be edited, file path to be imported
    def __init__(self,cred_service,sheet_name,file_path):

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(cred_service, scope)
        self.client = gspread.authorize(self.creds)
        self.sheet_name = sheet_name
        self.file_path = file_path

    # Import dataframe from textfile to individual google sheet (store head count each day)
    # Individual google sheet stores the dataframe and chart plot with the dataframe
    def dfImport(self):
        try:
            spreadsheet = self.client.open(self.sheet_name)

            worksheet = spreadsheet.sheet1

            # Read CSV into a Pandas DataFrame
            df = pd.read_csv(self.file_path,delimiter='\s',header=None,engine='python')

            # Define column name
            df.columns = ['Date','Time','Count']
            
            # Change middle column(time) to hour hand only
            df_hr = pd.to_datetime(df['Time'].to_list(),format='%H:%M:%S.%f').hour
            df['Time'] = df_hr

            dataUpdated = [df.columns.tolist()] + df.values.tolist()
            # Clear existing data in the Google Sheet
            worksheet.clear()

            # Write the DataFrame to the Google Sheet
            worksheet.update(dataUpdated)

            print(f"CSV data has been imported into {self.sheet_name} in Google Sheets.")
        except gspread.SpreadsheetNotFound:
            print(f"The spreadsheet '{self.sheet_name}' does not exist.")

        except Exception as e:
            print(f"An error occurred: (e)")        
    
    # Create chart for the google sheet (once only)
    def dfChart(self , sheet_name , fixed_Chart_ID , label_Cell ,x_axis_title ,count_Cell , count_cell_2 = 0 , count_cell_3 = 0 ):
        try:
            service_S = build("sheets","v4", credentials=self.creds)
            service_D = build("drive","v3",credentials=self.creds)

            # Check if the spreadsheet existed or not
            q_info = "name ='" + sheet_name + "'and mimeType = 'application/vnd.google-apps.spreadsheet'"
        
            txt_response = service_D.files().list(q= q_info,
                                                spaces = 'drive'
                                                ).execute()

            if txt_response['files']:

                spreadsheet = self.client.open(sheet_name)
                spreadsheet_ID = spreadsheet.id
                sheet_ID = spreadsheet.sheet1.id

                try:
                    # for checking if chart existed, if so skip this
                    target_sheet = spreadsheet.get_worksheet_by_id(fixed_Chart_ID)
                    print(f"The sheet with ID '{fixed_Chart_ID}' exists, proceed to skip this function")

                except gspread.WorksheetNotFound:
                    # if not exist, proceed with creating chart
                    print(f"Sheet with ID '{fixed_Chart_ID}' not found in the spreadsheet,proceed with chart bar creation")
                    
                    # check if user define col 2 and 3 range or not
                    if count_cell_2 or count_cell_3 == 0:

                        # Chart for one variable only
############################################################################################################################################
                        
                        request_body = {
                                    'requests': [
                                        {
                                        'addChart': 
                                        {
                                            'chart': 
                                            {
                                            'spec': 
                                            {
                                                'title': f"Accumulate head count as per {x_axis_title}",
                                                'basicChart': 
                                                {
                                                'chartType': "COLUMN",
                                                'legendPosition': "BOTTOM_LEGEND",
                                                'axis': 
                                                [
                                                    #X-axis
                                                    {
                                                    'position': "BOTTOM_AXIS",
                                                    'title': x_axis_title
                                                    },
                                                    #Y-axis
                                                    {
                                                    'position': "LEFT_AXIS",
                                                    'title': "Count"
                                                    }
                                                ],

                                                # Chart Label
                                                'domains': 
                                                [
                                                    {
                                                        'domain':
                                                        {
                                                            'sourceRange':
                                                            {
                                                                'sources':
                                                                [
                                                                    {
                                                                        'sheetId': sheet_ID,
                                                                        'startRowIndex': 0,
                                                                        'endRowIndex': 1000,
                                                                        'startColumnIndex': label_Cell,
                                                                        'endColumnIndex': label_Cell+1
                                                                    }
                                                                ]
                                                            }
                                                        }
                                                    }
                                                ],

                                                # Chart Data
                                                'series': 
                                                [
                                                    {
                                                    'series': 
                                                    {
                                                        'sourceRange': 
                                                        {
                                                        'sources': 
                                                        [
                                                            {
                                                            'sheetId': sheet_ID,
                                                            'startRowIndex': 0,
                                                            'endRowIndex': 1000,
                                                            'startColumnIndex': count_Cell,
                                                            'endColumnIndex': count_Cell+1
                                                            }
                                                        ]
                                                        }
                                                    },
                                                    'targetAxis': "LEFT_AXIS"
                                                    }
                                                ],
                                                'headerCount': 1
                                                }
                                            },
                                            'position': {
                                                "sheetId": fixed_Chart_ID
                                            }
                                            }
                                        }
                                        }
                                    ]
                                    }
                        
############################################################################################################################################

                    else:

                        # Chart for 3 variable
############################################################################################################################################

                        request_body = {
                                    'requests': [
                                        {
                                        'addChart': 
                                        {
                                            'chart': 
                                            {
                                            'spec': 
                                            {
                                                'title': f"Accumulate head count as per {x_axis_title}",
                                                'basicChart': 
                                                {
                                                'chartType': "COLUMN",
                                                'legendPosition': "BOTTOM_LEGEND",
                                                'axis': 
                                                [
                                                    #X-axis
                                                    {
                                                    'position': "BOTTOM_AXIS",
                                                    'title': x_axis_title
                                                    },
                                                    #Y-axis
                                                    {
                                                    'position': "LEFT_AXIS",
                                                    'title': "Count"
                                                    }
                                                ],

                                                # Chart Label
                                                'domains': 
                                                [
                                                    {
                                                        'domain':
                                                        {
                                                            'sourceRange':
                                                            {
                                                                'sources':
                                                                [
                                                                    {
                                                                        'sheetId': sheet_ID,
                                                                        'startRowIndex': 0,
                                                                        'endRowIndex': 1000,
                                                                        'startColumnIndex': label_Cell,
                                                                        'endColumnIndex': label_Cell+1
                                                                    }
                                                                ]
                                                            }
                                                        }
                                                    }
                                                ],

                                                # Chart Data
                                                'series': 
                                                [
                                                    # Col 1 (Reading corner 1[maybe?])
                                                    {
                                                    'series': 
                                                    {
                                                        'sourceRange': 
                                                        {
                                                        'sources': 
                                                        [
                                                            {
                                                            'sheetId': sheet_ID,
                                                            'startRowIndex': 0,
                                                            'endRowIndex': 1000,
                                                            'startColumnIndex': count_Cell,
                                                            'endColumnIndex': count_Cell+1
                                                            }
                                                        ]
                                                        }
                                                    },
                                                    'targetAxis': "LEFT_AXIS"
                                                    },

                                                    # Col 2 (Reading corner 2 cam 1[maybe?])
                                                    {
                                                    'series': 
                                                    {
                                                        'sourceRange': 
                                                        {
                                                        'sources': 
                                                        [
                                                            {
                                                            'sheetId': sheet_ID,
                                                            'startRowIndex': 0,
                                                            'endRowIndex': 1000,
                                                            'startColumnIndex': count_cell_2,
                                                            'endColumnIndex': count_cell_2 +1
                                                            }
                                                        ]
                                                        }
                                                    },
                                                    'targetAxis': "LEFT_AXIS"
                                                    },

                                                     # Col 3 (Reading corner 2 cam 2[maybe?])
                                                    {
                                                    'series': 
                                                    {
                                                        'sourceRange': 
                                                        {
                                                        'sources': 
                                                        [
                                                            {
                                                            'sheetId': sheet_ID,
                                                            'startRowIndex': 0,
                                                            'endRowIndex': 1000,
                                                            'startColumnIndex': count_cell_3,
                                                            'endColumnIndex': count_cell_3 +1
                                                            }
                                                        ]
                                                        }
                                                    },
                                                    'targetAxis': "LEFT_AXIS"
                                                    }
                                                ],
                                                'headerCount': 1
                                                }
                                            },
                                            'position': {
                                                "sheetId": fixed_Chart_ID
                                            }
                                            }
                                        }
                                        }
                                    ]
                                    }

############################################################################################################################################

                    response = service_S.spreadsheets().batchUpdate(
                        spreadsheetId = spreadsheet_ID,
                        body = request_body
                    ).execute()
                    print("chart created")
            else:
                print("cannot find file")

        except HttpError as e:
            print("Error: " + str(e))

    # Append to master file, 
    # For now will take in the last row data of one individual sheet (accumulated count) and append it to the master file
    def dfAppend(self,master_dfName):
        # Open individual file and get last 2 row data
        spreadsheet = self.client.open(self.sheet_name)
        worksheet   = spreadsheet.sheet1        
        try:
            length = len(worksheet.get_all_values())
            date        = worksheet.row_values(length-1)[0]
            latestCount = worksheet.row_values(length)[2]
        except:
            date        = worksheet.row_values(1)[0]
            latestCount = 0
        # combine the data
        latestData = [date, latestCount]
        
        # open master file
        spreadsheet = self.client.open(master_dfName)
        worksheet = spreadsheet.sheet1

        # Write column name to first column if spreadsheet is empty, will not write if there is data in other cell
        if worksheet.get_all_values() == []:
            worksheet.append_row(['date','count'],value_input_option='USER_ENTERED')

        worksheet.append_row(latestData,value_input_option='USER_ENTERED')
        
