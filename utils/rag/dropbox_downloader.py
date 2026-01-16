from typing import List, Tuple
import dropbox
from dropbox.exceptions import ApiError, AuthError
import requests
import zipfile
import io
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


class DropboxDownloader:
    """
    Class to list and download files from a publicly available Dropbox folder without API key.
    
    Uses the official Dropbox SDK where possible, with fallback to direct URL access
    for public shared links that don't require authentication.
    
    Supports both old format (https://www.dropbox.com/sh/xxxxx/yyyyy) and new format
    (https://www.dropbox.com/scl/fo/xxxxx/yyyyy?rlkey=...&st=...) shared links.
    """
    
    def __init__(self, shared_link: str):
        """
        Initialize Dropbox Downloader.
        
        Args:
            shared_link: The shared link URL of the Dropbox folder (e.g., 
                        https://www.dropbox.com/sh/xxxxx/yyyyy or
                        https://www.dropbox.com/scl/fo/xxxxx/yyyyy?rlkey=...&st=...&dl=0)
        """
        # Normalize the shared link but preserve query parameters
        # New Dropbox format requires rlkey and other params
        parsed = urlparse(shared_link.rstrip('/'))
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # Remove dl parameter if present (we'll add it when needed)
        if 'dl' in query_params:
            del query_params['dl']
        
        # Reconstruct URL without dl parameter
        # Keep all other query parameters (like rlkey, st, etc.)
        new_query = urlencode(query_params, doseq=True)
        self.shared_link = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        )).rstrip('/')
        
        # Store base URL and query params separately for easier manipulation
        self._base_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            '',
            ''
        )).rstrip('/')
        self._query_params = query_params
        
        # Try to initialize Dropbox client
        # Note: Most SDK operations require auth, but we'll try without first
        try:
            # Initialize with empty token - this might work for some public operations
            self.dbx = dropbox.Dropbox(oauth2_access_token="")
        except Exception:
            self.dbx = None
        
        # Cache for zip content to avoid re-downloading
        self._zip_cache = None
        self._file_list_cache = None
    
    def list_files(self) -> List[Tuple[str, str]]:
        """
        List all files from the configured public Dropbox folder.
        
        Returns:
            List of tuples containing (filename, file_path) for each file in the folder
        """
        results = []
        
        # Try using Dropbox SDK first
        if self.dbx:
            try:
                # Try to get shared link metadata
                shared_link_metadata = self.dbx.sharing_get_shared_link_metadata(
                    url=self.shared_link
                )
                
                # Check if it's a folder
                if isinstance(shared_link_metadata, dropbox.sharing.FolderLinkMetadata):
                    # Try to list folder contents using shared link
                    try:
                        folder_list_result = self.dbx.files_list_folder(
                            path="",
                            shared_link=dropbox.files.SharedLink(url=self.shared_link)
                        )
                        
                        entries = folder_list_result.entries
                        
                        # Handle pagination
                        while folder_list_result.has_more:
                            folder_list_result = self.dbx.files_list_folder_continue(
                                cursor=folder_list_result.cursor
                            )
                            entries.extend(folder_list_result.entries)
                        
                        # Extract file information
                        for entry in entries:
                            if isinstance(entry, dropbox.files.FileMetadata):
                                file_path = entry.path_display or entry.name
                                results.append((entry.name, file_path))
                        
                        return results
                    except (ApiError, AuthError):
                        # SDK method failed, fall through to URL-based method
                        pass
                        
            except (ApiError, AuthError):
                # SDK requires authentication, fall through to URL-based method
                pass
        
        # Fallback: For public folders without auth, download as zip and extract file list
        # This is the only way to list files in a public folder without authentication
        try:
            # Use cached file list if available
            if self._file_list_cache is not None:
                return self._file_list_cache
            
            # Download the folder as a zip file (or use cache)
            if self._zip_cache is None:
                # Construct download URL with dl=1 while preserving other query params
                zip_query_params = self._query_params.copy()
                zip_query_params['dl'] = ['1']
                zip_query = urlencode(zip_query_params, doseq=True)
                zip_url = f"{self._base_url}?{zip_query}"
                
                response = requests.get(zip_url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
                response.raise_for_status()
                self._zip_cache = io.BytesIO(response.content)
            
            # Reset zip file pointer
            self._zip_cache.seek(0)
            
            # Extract file list from zip
            with zipfile.ZipFile(self._zip_cache, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Filter out directories and get file names
                for file_name in file_list:
                    if not file_name.endswith('/'):  # Skip directories
                        filename = file_name.split('/')[-1]
                        # Store the file name and use it as the path for download
                        results.append((filename, file_name))
            
            # Cache the results
            self._file_list_cache = results
            return results
            
        except Exception:
            # If everything fails, return empty list
            return []
    
    def download_file(self, file_path: str) -> bytes:
        """
        Download a file from Dropbox to memory based on file path.
        
        Args:
            file_path: The path of the file to download (from list_files)
            
        Returns:
            File content as bytes
            
        Raises:
            ValueError: If the file cannot be downloaded
        """
        # Try using Dropbox SDK first
        if self.dbx:
            try:
                # Use files_download with shared link
                metadata, response = self.dbx.files_download(
                    path=file_path,
                    shared_link=dropbox.files.SharedLink(url=self.shared_link)
                )
                
                return response.content
                
            except (ApiError, AuthError):
                # SDK method failed, fall through to URL-based method
                pass
        
        # Fallback: For public folders, we need to download the zip and extract the file
        # This is because individual file links aren't available without auth
        try:
            # Use cached zip if available, otherwise download
            if self._zip_cache is None:
                # Construct download URL with dl=1 while preserving other query params
                zip_query_params = self._query_params.copy()
                zip_query_params['dl'] = ['1']
                zip_query = urlencode(zip_query_params, doseq=True)
                zip_url = f"{self._base_url}?{zip_query}"
                
                response = requests.get(zip_url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
                response.raise_for_status()
                self._zip_cache = io.BytesIO(response.content)
            
            # Reset zip file pointer
            self._zip_cache.seek(0)
            
            # Extract the specific file
            with zipfile.ZipFile(self._zip_cache, 'r') as zip_ref:
                # Normalize the file path (remove leading slash if present)
                normalized_path = file_path.lstrip('/')
                
                # Try to find the file in the zip
                if normalized_path in zip_ref.namelist():
                    return zip_ref.read(normalized_path)
                else:
                    # Try to find by filename only
                    for name in zip_ref.namelist():
                        if name.endswith(file_path) or name.split('/')[-1] == file_path:
                            return zip_ref.read(name)
                    
                    raise ValueError(f"File {file_path} not found in zip")
            
        except Exception as e:
            raise ValueError(f"Could not download file: {e}")
