import os
import sharepy
from app.utils.global_memory import GlobalMemoryClass
import json


class SharePointInteractionClass:

    def __init__(self):
        self.logger = GlobalMemoryClass.LOGGER

    def download_from_sharepoint(self, settings):

        br_site_name = settings['br_site_name']
        site_name = settings['site_name']
        relative_folder = settings['relative_folder']
        upload_folder = settings['upload_folder']
        local_folder_path = settings['local_folder_path']

        self.logger.info("Downloading all files in the following relative folder: {}".format(relative_folder))
        self.logger.info("Logging into sharepoint")
        s = sharepy.connect("https://broadreachcorp.sharepoint.com",
                            username=settings['user_credentials']['username'],
                            password=settings['user_credentials']['password'])

        self.logger.info("Get all files in relative folder")
        response = s.get(f"https://{br_site_name}/sites/{site_name}/_api/web/GetFolderByServerRelativeUrl('{relative_folder}')/Files")
        json_data = json.loads(response.text)['d']['results']

        if not os.path.exists('{}/{}'.format(local_folder_path, upload_folder)):
            os.makedirs('{}/{}'.format(local_folder_path, upload_folder))

        file_dict_list = []
        for file in json_data:
            time_created = file['TimeCreated']
            file_url = file['ServerRelativeUrl']
            file_name = file['Name']
            self.logger.info("Downloading the folllowing file: {}".format(file_url))
            self.logger.info("- that was created on: {}".format(time_created))
            s.getfile(f"https://{br_site_name}/{file_url}", filename='{}/{}/{}'.format(local_folder_path, upload_folder, file_name))
            file_dict_list.append(dict(file_name=file_name, file_url=file_url, date=time_created))
        return file_dict_list

    def upload_to_sharepoint(self, settings):

        self.logger.info("-configure sharepoint settings")
        folder_path = settings['folder_path']
        sp_folder_name = settings['sp_folder_name']
        sp_url = settings['sp_url']
        site_name = settings['site_name']
        library_name = settings['library_name']
        files_to_upload = settings['files_to_upload']

        self.logger.info("-check is the session files exists")
        # check is the session files exists
        if os.path.isfile("sp-session.pkl"):
            s = sharepy.load()
        else:
            s = sharepy.connect("https://broadreachcorp.sharepoint.com",
                                username=settings['user_credentials']['username'],
                                password=settings['user_credentials']['password'])
            s.save()

        self.logger.info("-check is system is windows")
        # check is system is windows
        if os.name == 'nt':
            folder = folder_path.split('\\')
        else:
            folder = folder_path.split('/')

        self.logger.info("--check to see if the folder_path is a directory")
        # check to see if the folder_path is a directory
        if os.path.isdir(folder_path):

            self.logger.info("-creates the folder in sharepoint")
            # creates the folder in sharepoint
            s.post("https://" + sp_url + "/sites/" + site_name + "/_api/web/folders",
                   json={
                       "__metadata": {"type": "SP.Folder"},
                       "ServerRelativeUrl": '{}/{}'.format(library_name, sp_folder_name)
                   })

            self.logger.info("-uploads files to sharepoint")
            # uploads files to sharepoint
            for file_upload in files_to_upload:
                headers = {"accept": "application/json;odata=verbose",
                           "content-type": "application/x-www-urlencoded; charset=UTF-8"}

                with open(os.path.join(folder_path, file_upload), 'rb') as read_file:
                    content = read_file.read()

                self.logger.info("post csv file")
                s.post(f"https://{sp_url}/sites/{site_name}/_api/web/GetFolderByServerRelativeUrl('{library_name}/{sp_folder_name}')/Files/add(url='{file_upload}',overwrite=true)",
                       data=content, headers=headers)
                self.logger.info("done with post of csv file")
                self.logger.info(folder)
