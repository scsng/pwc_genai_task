from typing import List, Tuple
import requests
import zipfile
import io
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


class DropboxDownloader:
    """Downloads files from public Dropbox folders without authentication.
    
    Downloads the folder as a zip file and extracts individual files from it.
    Supports both legacy (sh/) and new (scl/fo/) Dropbox shared link formats.
    """
    
    def __init__(self, shared_link: str):
        """Initialize the downloader with a Dropbox shared folder link.
        
        Args:
            shared_link: Public shared link to a Dropbox folder. Supports formats:
                - Legacy: https://www.dropbox.com/sh/xxxxx/yyyyy
                - New: https://www.dropbox.com/scl/fo/xxxxx/yyyyy?rlkey=...&st=...
        """
        self.shared_link, self._base_url, self._query_params = self._normalize_url(shared_link)
        self._zip_cache = None
        self._file_list_cache = None
    
    def _normalize_url(self, shared_link: str) -> Tuple[str, str, dict]:
        """Normalize Dropbox shared link and extract components.
        
        Args:
            shared_link: Raw shared link URL.
            
        Returns:
            Tuple of (normalized_link, base_url, query_params).
        """
        parsed = urlparse(shared_link.rstrip('/'))
        query_params = parse_qs(parsed.query)
        
        if 'dl' in query_params:
            del query_params['dl']
        
        new_query = urlencode(query_params, doseq=True)
        normalized_link = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        )).rstrip('/')
        
        base_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            '',
            ''
        )).rstrip('/')
        
        return normalized_link, base_url, query_params
    
    def list_files(self) -> List[Tuple[str, str]]:
        """List all files in the configured Dropbox folder.
        
        Downloads the folder as a zip and extracts the file list from it.
        
        Returns:
            List of (filename, file_path) tuples for each file in the folder.
        """
        if self._file_list_cache is not None:
            return self._file_list_cache
        
        try:
            self._ensure_zip_cache()
            self._zip_cache.seek(0)
            
            results = []
            with zipfile.ZipFile(self._zip_cache, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if not file_name.endswith('/'):
                        filename = file_name.split('/')[-1]
                        results.append((filename, file_name))
            
            self._file_list_cache = results
            return results
            
        except Exception:
            return []
    
    def download_file(self, file_path: str) -> bytes:
        """Download a file from Dropbox to memory by extracting from zip.
        
        Args:
            file_path: Path of the file to download (from list_files).
            
        Returns:
            File content as bytes.
            
        Raises:
            ValueError: If the file cannot be downloaded or found.
        """
        try:
            self._ensure_zip_cache()
            self._zip_cache.seek(0)
            return self._extract_file_from_zip(file_path)
        except Exception as e:
            raise ValueError(f"Could not download file: {e}")
    
    def _ensure_zip_cache(self) -> None:
        """Download and cache the folder as a zip file if not already cached."""
        if self._zip_cache is None:
            zip_query_params = self._query_params.copy()
            zip_query_params['dl'] = ['1']
            zip_query = urlencode(zip_query_params, doseq=True)
            zip_url = f"{self._base_url}?{zip_query}"
            
            response = requests.get(zip_url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
            response.raise_for_status()
            self._zip_cache = io.BytesIO(response.content)
    
    def _extract_file_from_zip(self, file_path: str) -> bytes:
        """Extract a specific file from the cached zip.
        
        Args:
            file_path: Path of the file to extract.
            
        Returns:
            File content as bytes.
            
        Raises:
            ValueError: If the file is not found in the zip.
        """
        normalized_path = file_path.lstrip('/')
        
        with zipfile.ZipFile(self._zip_cache, 'r') as zip_ref:
            if normalized_path in zip_ref.namelist():
                return zip_ref.read(normalized_path)
            
            for name in zip_ref.namelist():
                if name.endswith(file_path) or name.split('/')[-1] == file_path:
                    return zip_ref.read(name)
            
            raise ValueError(f"File {file_path} not found in zip")
