# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import pickle
import time

try:
    import librosa
    import requests
    import requests_oauthlib
    from joblib import Parallel, delayed
    from oauthlib.oauth2 import TokenExpiredError
except (ModuleNotFoundError, ImportError) as e:
    raise e

try:
    import freesound
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "freesound is not installed. Execute `pip install --no-cache-dir git+https://github.com/MTG/freesound-python.git` in terminal"
    )


"""
Instructions
1. We will need some requirements including freesound, requests, requests_oauthlib, joblib, librosa and sox. If they are not installed, please run `pip install -r freesound_requirements.txt`
2. Create an API key for freesound.org at  https://freesound.org/help/developers/
3. Create a python file called `freesound_private_apikey.py` and add lined `api_key = <your Freesound api key>` and `client_id = <your Freesound client id>`
4. Authorize by run `python freesound_download.py --authorize` and visit website, and paste response code
5. Feel free to change any arguments in download_resample_freesound.sh such as max_samples and max_filesize
6. Run `bash download_resample_freesound.sh <numbers of files you want> <download data directory> <resampled data directory>`
"""

# Import the API Key
try:
    from freesound_private_apikey import api_key, client_id

    print("API Key found !")
except ImportError:
    raise ImportError(
        "Create a python file called `freesound_private_apikey.py` and add lined `api_key = <your Freesound api key>` and `client_id = <your Freesound client id>`"
    )

auth_url = 'https://freesound.org/apiv2/oauth2/authorize/'
redirect_url = 'https://freesound.org/home/app_permissions/permission_granted/'
token_url = 'https://freesound.org/apiv2/oauth2/access_token/'
scope = ["read", "write"]

BACKGROUND_CLASSES = [
    "Air brake",
    "Static",
    "Acoustic environment",
    "Distortion",
    "Tape hiss",
    "Hubbub",
    "Vibration",
    "Cacophony",
    "Throbbing",
    "Reverberation",
    "Inside, public space",
    "Inside, small room",
    "Echo",
    "Outside, rural",
    "Outside, natural",
    "Outside, urban",
    "Outside, manmade",
    "Car",
    "Bus",
    "Traffic noise",
    "Roadway noise",
    "Truck",
    "Emergency vehicle",
    "Motorcycle",
    "Aircraft engine",
    "Aircraft",
    "Helicopter",
    "Bicycle",
    "Skateboard",
    "Subway, metro, underground",
    "Railroad car",
    "Train wagon",
    "Train",
    "Sailboat",
    "Rowboat",
    "Ship",
]

SPEECH_CLASSES = [
    "Male speech",
    "Female speech",
    "Speech synthesizer",
    "Babbling",
    "Conversation",
    "Child speech",
    "Narration",
    "Laughter",
    "Yawn",
    "Whispering",
    "Whimper",
    "Baby cry",
    "Sigh",
    "Groan",
    "Humming",
    "Male singing",
    "Female singing",
    "Child singing",
    "Children shouting",
]


def initialize_oauth():
    # If token already exists, then just load it
    if os.path.exists('_token.pkl'):
        token = unpickle_object('_token')
        oauth = requests_oauthlib.OAuth2Session(client_id, redirect_uri=redirect_url, scope=scope, token=token)

    else:
        # Construct a new token after OAuth2 flow
        # Initialize a OAuth2 session
        oauth = requests_oauthlib.OAuth2Session(client_id, redirect_uri=redirect_url, scope=scope)

        authorization_url, state = oauth.authorization_url(auth_url)
        print(f"Visit below website and paste access token below : \n\n{authorization_url}\n")

        authorization_response = input("Paste authorization response code here :\n")

        token = oauth.fetch_token(
            token_url,
            authorization_response=authorization_response,
            code=authorization_response,
            client_secret=api_key,
        )

        # Save the token generated
        pickle_object(token, '_token')

    return oauth, token


def instantiate_session():
    # Reconstruct session in process, and force singular execution thread to reduce session
    # connections to server
    token = unpickle_object('_token')
    session = requests_oauthlib.OAuth2Session(client_id, redirect_uri=redirect_url, scope=scope, token=token)
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount('http://', adapter)
    return session


def refresh_token(session):
    print("Refreshing tokens...")
    # Token expired, perform token refresh
    extras = {'client_id': client_id, 'client_secret': api_key}
    token = session.refresh_token(token_url, **extras)
    print("Token refresh performed...")
    # Save the refreshed token
    pickle_object(token, '_token')
    return session


def pickle_object(token, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(token, f)


def unpickle_object(name):
    fp = name + '.pkl'
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            token = pickle.load(f)

        return token
    else:
        raise FileNotFoundError('Token not found!')


def is_resource_limited(e: freesound.FreesoundException):
    """
    Test if the reason for a freesound exception was either rate limit
    or daily limit.

    If it was for either reason, sleep for an appropriate delay and return
    to try again.

    Args:
        e: Freesound Exception object

    Returns:
        A boolean which describes whether the error was due to some
        api limit issue, or if it was some other reason.

        If false is returned, then the user should carefully check the cause
        and log it.
    """
    detail = e.detail['detail']

    if '2000' in detail:
        # This is the request limit, hold off for 1 hour and try again
        print(f"Hit daily limit, sleeping for 20 minutes.")
        time.sleep(60 * 20)
        return True

    elif '60' in detail:
        # This is the request limit per minute, hold off for 1 minute and try again
        print(f"Hit rate limit, sleeping for 1 minute.")
        time.sleep(60)
        return True

    else:
        return False


def prepare_client(client: freesound.FreesoundClient, token) -> freesound.FreesoundClient:
    # Initialize the client with token auth
    client.set_token(token['access_token'], auth_type='oauth')
    print("Client ready !")
    return client


def get_text_query_with_resource_limit_checks(client, query: str, filters: list, fields: str, page_size: int):
    """
    Performs a text query, checks for rate / api limits, and retries.

    Args:
        client: FreesoundAPI client
        query: query string (either exact or inexact)
        filters: list of string filters
        fields: String of values to recover
        page_size: samples per page returned

    Returns:

    """
    pages = None
    attempts = 20

    while pages is None:
        try:
            pages = client.text_search(query=query, filter=" ".join(filters), fields=fields, page_size=str(page_size),)

        except freesound.FreesoundException as e:
            # Most probably a rate limit or a request limit
            # Check if that was the case, and wait appropriate ammount of time
            # for retry
            was_resource_limited = is_resource_limited(e)

            # If result of test False, it means that failure was due to some other reason.
            # Log it, then break loop
            if not was_resource_limited:
                print(e.with_traceback(None))
                break

        attempts -= 1

        # Attempt to refresh tokens if it fails multiple times
        if attempts % 5 == 0 and attempts > 0:
            session = instantiate_session()
            refresh_token(session)
            session.close()
            token = unpickle_object('_token')
            client = prepare_client(client, token)

        if attempts <= 0:
            print(f"Failed to query pages for '{query}' after 10 attempts, skipping query")
            break

        if pages is None:
            print(f"Query attempts remaining = {attempts}")

    return client, pages


def get_resource_with_auto_refresh(session, download_url):
    """
    Attempts download of audio with a token refresh if necessary.
    """
    try:
        result = session.get(download_url)

    except TokenExpiredError as e:
        session = refresh_token(session)
        result = session.get(download_url)

    except Exception as e:
        result = None

        print(f"Skipping file {download_url} due to exception below\n\n")
        print(e)

    return result.content


def download_song(basepath, id, name, download_url):
    # Cleanup name
    name = name.encode('ascii', 'replace').decode()
    name = name.replace("?", "-")
    name = name.replace(":", "-")
    name = name.replace("(", "-")
    name = name.replace(")", "-")
    name = name.replace("'", "")
    name = name.replace(",", "-")
    name = name.replace("/", "-")
    name = name.replace("\\", "-")
    name = name.replace(".", "-")
    name = name.replace(" ", "")

    # Correct last `.` for filetype
    name = name[:-4] + '.wav'

    # Add file id to filename
    name = f"id_{id}" + "_" + name

    fp = os.path.join(basepath, name)

    # Check if file, if exists already, can be loaded by librosa
    # If it cannot be loaded, possibly corrupted file.
    # Delete and then re-download
    if os.path.exists(fp):
        try:
            _ = librosa.load(path=fp)
        except Exception:
            # File is currupted, delete and re-download.
            os.remove(fp)

            print(f"Pre-existing file {fp} was corrupt and was deleted, will be re-downloaded.")

    if not os.path.exists(fp):
        print("Downloading file :", name)

        session = instantiate_session()

        data = None
        attempts = 10

        try:
            while data is None:

                try:
                    # Get the sound data
                    data = get_resource_with_auto_refresh(session, download_url)

                except freesound.FreesoundException as e:
                    # Most probably a rate limit or a request limit
                    # Check if that was the case, and wait appropriate amount of time
                    # for retry
                    was_resource_limited = is_resource_limited(e)

                    # If result of test False, it means that failure was due to some other reason.
                    # Log it, then break loop
                    if not was_resource_limited:
                        print(e)
                        break

                attempts -= 1

                if attempts <= 0:
                    print(f"Failed to download file {fp} after 10 attempts, skipping file")
                    break

                if data is None:
                    print(f"Download attempts remaining = {attempts}")

        finally:
            session.close()

        # Write the data to file
        if data is not None:
            print("Downloaded file :", name)

            with open(fp, 'wb') as f:
                f.write(data)

            # If file size is less than 89, then this probably is a text format and not an actual audio file.
            if os.path.getsize(fp) > 89:
                print(f"File written : {fp}")

            else:
                os.remove(fp)
                print(f"File corrupted and has been deleted: {fp}")

        else:
            print(f"File [{fp}] corrupted or faced some issue when downloading, skipped.")

        # Sleep to avoid hitting rate limits
        time.sleep(5)

    else:
        print(f"File [{fp}] already exists in dataset, skipping re-download.")


def get_songs_by_category(
    client: freesound.FreesoundClient,
    category: str,
    data_dir: str,
    max_num_samples=100,
    page_size=100,
    min_filesize_in_mb=0,
    max_filesize_in_mb=10,
    n_jobs=None,
):
    """
    Download songs of a category with restrictions

    Args:
        client: FreesoundAPI client
        category: category to be downloaded
        data_dir: directory of downloaded songs
        max_num_samples: maximum number of samples of this category
        page_size: samples per page returned
        min_filesize_in_mb: minimum filesize of the song in MB
        max_filesize_in_mb: maximum filesize of the song in MB
        n_jobs: number of jobs for parallel processing

    Returns:

    """
    # quote string to force exact match
    query = f'"{category}"'
    print(f"Query : {query}")

    page_size = min(page_size, 150)
    max_filesize = int(max_filesize_in_mb * (2 ** 20))

    if min_filesize_in_mb == 0:
        min_filesize_in_mb = 1
    else:
        min_filesize_in_mb = int(min_filesize_in_mb * (2 ** 20))

    if max_num_samples < 0:
        max_num_samples = int(1e6)

    filters = [
        'type:(wav OR flac)',
        'license:("Attribution" OR "Creative Commons 0")',
        f'filesize:[{min_filesize_in_mb} TO {max_filesize}]',
    ]

    fields = "id,name,download,license"

    client, pages = get_text_query_with_resource_limit_checks(
        client, query=query, filters=filters, fields=fields, page_size=page_size
    )

    if pages is None:
        print(f"Number of attempts exceeded limit, skipping query {query}")
        return

    num_pages = pages.count

    # Check if returned empty result; if so, fallback to inexact category search
    if num_pages == 0:
        print(f"Found 0 samples of results for query '{query}'")
        print(f"Trying less restricted query : {category}")

        client, pages = get_text_query_with_resource_limit_checks(
            client, query=category, filters=filters, fields=fields, page_size=page_size
        )

        if pages is None:
            print(f"Number of attempts exceeded limit, skipping query {query}")
            return

        num_pages = pages.count

    print(f"Found {num_pages} samples of results for query '{query}'")

    category = category.replace(' ', '_')
    basepath = os.path.join(data_dir, category)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    sounds = []
    sample_count = 0

    # Retrieve sound license information
    with open(os.path.join(basepath, 'licenses.txt'), 'w') as f:
        f.write("ID,LICENSE\n")
        f.flush()

        while True:
            for sound in pages:
                if sample_count >= max_num_samples:
                    print(
                        f"Collected {sample_count} samples, which is >= max number of samples requested "
                        f"{max_num_samples}. Stopping for this category : {category}"
                    )
                    break

                sounds.append(sound)
                sample_count += 1

                f.write(f"{sound.id},{sound.license}\n")
                f.flush()

            if sample_count >= max_num_samples:
                break

            try:
                pages = pages.next_page()
            except ValueError:
                break

    if n_jobs is None:
        n_jobs = max(1, len(sounds))

    # Parallel download all songs
    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        _ = parallel(delayed(download_song)(basepath, sound.id, sound.name, sound.download) for sound in sounds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Freesound download script")

    parser.add_argument(
        '--authorize', action='store_true', dest='auth', help='Flag to only perform OAuth2 authorization step'
    )

    parser.add_argument('-c', '--category', default='', type=str, help='Category required to download')

    parser.add_argument('-d', '--data_dir', default='', type=str, help='Destination folder to store data')

    parser.add_argument('--page_size', default=100, type=int, help='Number of sounds per page')

    parser.add_argument('--max_samples', default=100, type=int, help='Maximum number of sound samples')

    parser.add_argument('--min_filesize', default=0, type=int, help='Maximum filesize allowed (in MB)')

    parser.add_argument('--max_filesize', default=20, type=int, help='Maximum filesize allowed (in MB)')

    parser.set_defaults(auth=False)

    args = parser.parse_args()

    if args.auth:
        """ Initialize oauth token to be used by all """
        oauth, token = initialize_oauth()
        oauth.close()

        print("Authentication suceeded ! Token stored in `_token.pkl`")
        exit(0)

    if not os.path.exists('_token.pkl'):
        raise FileNotFoundError(
            "Please authorize the application first using " "`python freesound_download.py --authorize`"
        )
    if args.data_dir == '':
        raise ValueError("Data dir must be passed as an argument using `--data_dir`")

    data_dir = args.data_dir

    page_size = args.page_size
    max_num_samples = args.max_samples
    min_filesize_in_mb = args.min_filesize
    max_filesize_in_mb = args.max_filesize

    # Initialize and authenticate client
    token = unpickle_object('_token')
    freesound_client = freesound.FreesoundClient()
    client = prepare_client(freesound_client, token)

    category = args.category

    if category == '':
        raise ValueError("Cannot pass empty string as it will select all of FreeSound data !")

    print(f"Downloading category : {category}")
    get_songs_by_category(
        client,
        category,
        data_dir=data_dir,
        max_num_samples=max_num_samples,
        page_size=page_size,
        min_filesize_in_mb=min_filesize_in_mb,
        max_filesize_in_mb=max_filesize_in_mb,
        n_jobs=30,
    )
