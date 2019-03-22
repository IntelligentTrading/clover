"""
Django settings for apps project.

Generated by 'django-admin startproject' using Django 1.9.9.

For more information on this file, see
https://docs.djangoproject.com/en/1.9/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.9/ref/settings/
"""
import os
import sys
import logging


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


# Add the local vendor libs path
sys.path.append(BASE_DIR + "/vendorlibs")


# DEFINE THE ENVIRONMENT TYPE
PRODUCTION = STAGE = DEMO = LOCAL = False
DEPLOYMENT_TYPE = os.environ.get('DEPLOYMENT_TYPE', 'LOCAL').upper()

PRODUCTION = 'PRODUCTION' in DEPLOYMENT_TYPE
DEMO = 'DEMO' in DEPLOYMENT_TYPE
STAGE = 'STAGE' in DEPLOYMENT_TYPE
LOCAL = 'LOCAL' in DEPLOYMENT_TYPE


# for Heroku deployment wierdness
LOAD_TALIB = True

# Set up logger
if LOCAL:
    log_level = logging.DEBUG
elif PRODUCTION:
    log_level = logging.INFO
elif STAGE:
    log_level = logging.DEBUG
else:
    log_level = logging.DEBUG

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

if LOCAL or STAGE:
    logging.getLogger('boto').setLevel(logging.INFO)
    logging.getLogger('pyqs').setLevel(logging.INFO)

logger.info("Deployment environment detected: {}".format(DEPLOYMENT_TYPE))


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', '')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = not PRODUCTION

ALLOWED_HOSTS = [
    '.herokuapp.com',
    'localhost',
    '127.0.0.1',
    '.ngrok.io'
]


# Application definition
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize'
]

THIRD_PARTY_APPS = [
    'django_extensions',
]

LOCAL_APPS = [
    # 'apps.user',
    'apps.common',
    'apps.communication',
    'apps.TA',
    'apps.portfolio',
    'apps.doge',
    'apps.tests'
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware', # for static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'settings.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [os.path.join(SITE_ROOT, 'templates'), ],
    'APP_DIRS': True,
    'OPTIONS': {
        'context_processors': [
            'django.template.context_processors.debug',
            'django.template.context_processors.request',
            'django.template.context_processors.media',
            'django.contrib.auth.context_processors.auth',
            'django.template.context_processors.static',
            'django.contrib.messages.context_processors.messages',
        ],
    },
}, ]

LOGIN_REDIRECT_URL = '/portfolio/'

WSGI_APPLICATION = 'settings.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.9/ref/settings/#databases
# DATABASES = {}


# Password validation
# https://docs.djangoproject.com/en/1.9/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.9/topics/i18n/

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = False
USE_L10N = False
USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.9/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(SITE_ROOT, 'staticfiles')
# Additional locations of static files
STATICFILES_DIRS = (
    os.path.join(SITE_ROOT, 'static'),
)
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

try:
    from settings.vendor_settings import *
except Exception as e:
    logger.warning("Failed to import vendor_services_settings.")
    logger.warning(str(e))


if LOCAL:
    logger.info("LOCAL environment detected. Importing local_settings.py")
    try:
        from settings.local_settings import *
    except:
        logger.error("Could not successfully import local_settings.py. This is necessary if you are running locally. This file should be in version control.")
        raise

# mapping from bin size to a name short/medium
# CHANGES TO THESE VALUES REQUIRE MAKING AND RUNNING DB MIGRATIONS
PERIODS_LIST = list([60,240,1440])  # minutes, so 1hr, 4hr, 24hr
# CHANGES TO THESE VALUES REQUIRE MAKING AND RUNNING DB MIGRATIONS
(SHORT, MEDIUM, LONG) = PERIODS_LIST
HORIZONS_TIME2NAMES = {
    SHORT:'short',
    MEDIUM:'medium',
    LONG:'long'
}

# list of the exchanges on which we generate signals. Make it in sync with same list in Data app settings
#EXCHANGE_MARKETS = ('poloniex', 'binance', 'bittrex', 'bitfinex', 'kucoin')
EXCHANGE_MARKETS = ('poloniex', 'binance', 'bittrex')

LOAD_TALIB = True

# list of tickers for which doges will vote
SUPPORTED_DOGE_TICKERS = ['BTC_USDT', 'ETH_USDT', ]

ONE_WEEK = 60*60*24*7
ONE_DAY = 60*60*24
ONE_HOUR = 60*60

# doge training schedule (how often to retrain and reinit the committee)
DOGE_RETRAINING_PERIOD_SECONDS = ONE_HOUR           # how often to retrain and reinit the committee
DOGE_TRAINING_PERIOD_DURATION_SECONDS = ONE_DAY     # the duration of the training period
DOGE_LOAD_ROCKSTARS = True
DOGE_MAX_ROCKSTARS = 20
DOGE_REBALANCING_PERIOD_SECONDS = 20*60             # how often to run the rebalancer when autotrading