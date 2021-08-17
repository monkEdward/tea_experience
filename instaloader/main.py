# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time
from datetime import datetime
from os import walk
from pathlib import Path

from instaloader import instaloader


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


class MyRateController(instaloader.RateController):
    def sleep(self, secs):
        print('avvio applicazione in sleep')
        time.sleep(secs)
        print('rilancio applicazione')
        init()
        raise Exception()


def init():
    L = instaloader.Instaloader(download_videos=False, download_video_thumbnails=False, compress_json=False,
                                save_metadata=False, rate_controller=lambda ctx: MyRateController(ctx))
    L.login('', '')
    sellers = ['teasnsuch', 'woashwellness', 'shopteappo', 'alldayteaclub', 'tejanateas', 'teapigs', 'dualinnovation_nl',
               'natureinyou.eu', 'thejadeleaf', 'drinkreddiamond', 'tapalofficial', 'sdcoffeetea', 'myteabrk',
               'drinkmilos', 'mightyleaftea', 'tavalontea', 'tazo', 'waghbakritea.official', 'zarutea', 'yogitea',
               'typhootea', 'redsparrowteacompany', 'kissatea', 'calmersutratea_ny', 'tea_blossoms', 'teashop.aus',
               'myteagirl.au', 'teablendsforyou', 'teaandscandal', 'teapig', 'societytea', 'teabar', 'teaplusdrinks',
               'teacultureoftheworld', 'potheadstea', 'ketepalimited', 'teabeyond', 'teasatgirnar', 'brewhousetea',
               'drinkteaindia', 'twrlmilktea', 'tchaba_arabia', 'melsteas', 'secretsoftea', 'tamateaco', 'teacotr',
               'teabiotics', 'drinkcusa', 'meleztea', 'teavalleytea', 'teafromthemanor', 'avocadoleaftea',
               'greattearoad', 'theteatrove', 'marktwendelltea', 'teashopcafe', 'dharmsalateaco', 'samovartea',
               'wildorchardgreente>a', 'tuskertea', 'theorganicteaproject', 'teathief_teas', 'goldenwattletea',
               'sydneytea1', 'elmstocktea', 'pekoebrew', 'teatribe', 'parlortea', 'wanlingteahouse',
               'house_of_tea_and_coffee', 'bodhiorganictea', 'rainbowchaitea', 'monji_tea', 'the.tea.collective',
               'theberryteashop', 'cleantea', 'sinensisaus', 'teapressaus', 'austeashop', 'teaguy11', 'altitudetea',
               'redleaftea_au', 'tamborinetea', 'GossipTea.au', 'parroteyestea', 'hotteamama', 'kkteabd',
               'ispahani_mirzapore_tea', 'tealeafpro', 'lilistea1', 'beyourtea', 'xantea_organic', 'releasetea',
               'essentials_tea', 'missesteaspoon', 'tea_botanical', 'regen_tea', 'teaandco_br', 'teaetox', 'think.tea',
               'gan.tea.studio', 'teacomdu', 'telice.tea', 'masters_tea_and_coffee', 'sloanetea', 'taotealeaf',
               'spiretea', 'millennia_tea', 'shissotea', 'groundedtea', 'myteabot', 'shantitea', 'blueocean_tea',
               'teahaus', 'greystonetea', 'powteaco', 'mettateaco', 'twohillstea', 'teajaoffice', 'superbolt.tea',
               'teamonkeyofficial', 'denmantea.ca', 'divineorganictea', 'bluemountainteaco', 'baddadtea',
               'teahorseteas', 'cjaytea', 'virtue_tea', 'canadianbarleyteacompany', 'sipsby', 'zeylantea',
               'blissteakombucha', 'bellasabatina', 'lifeboostertea', 'lotatea', 'officialmylifetea',
               'austinenglishtea', 'blenda_teas', 'catateacom', 'symphonyofleavestea', 'twospoonstea', 'boulderteaco',
               'theteapickerofficial', 'lovetheteathief', 'wellteauk', 'rochellesteaspa', 'blumintea',
               'shine_tea_agency_', 'get2steeping', 'notsogentletea', 'windsorteaemporium', 'naoteas', 'teawithherbs',
               'miladystea', 'mikas.tea', 'real_thesaurustea', 'ellys.tea', 'molbaniteaco', 'myteaquility',
               'teassentialteas', 'vedawellnessteas', 'mycha.tea', 'merakiartisanteas', 'teasakao', 'andestea.cl',
               'jamaicatea', 'magictea_chile', 'goodteacl', 'runmingteas', 'orientalteacn', 'bioteacn', 'aiyamatcha',
               'shinewing', 'asiantea_ita', 'alvitateas', 'azercay.az', 'beautifultaiwanteacompany', 'birdpicktea',
               'brewnationtea', 'cainatea', 'timelessenergyforlife', '_teaistheanswer', 'aceteatw', 'dachi_tea_co',
               'yamadaitea', 'riyang_teayard', 'yinchen_teapot', 'yiqinteahouse', 'teaheaduk', 'moychay.nvrsk',
               'puretea.info', 'brewedleaf', 'yxteahouse', 'teatastingbymingcha', 'theteaporter', 'hencetea',
               'toptreeherbs_', 'hugotea.space', 'haatea_official', 'utopictea', 'kapateecpt', '_tea4me_', '180andup',
               'chai_walli', 'sherpachai', 'tastychai', 'onestripechaico', 'chaicraftindia', 'drinkmechai',
               'chaichuntea', 'teaavenue1936', 'theteacompany', 'bogawantalawatea1869', 'cultivatetastetea',
               'jenweytea', 'bighearttea', 'royalteacoffeeco_clearlake', 'vikramtea', 'ustwotea', 'samaaratea',
               'moncloateaboutique', 'suntipstea', 'teaspressa', 'verdanttea', 'teapeopleus', 'slteaboard', 'luxmitea',
               'the_teacompany', 'pmdtea', 'teaplusus', 'premierstea_korea', 'etsteas', 'glenburntea', 'rungtatea',
               'dobrateame', 'jivraj9tea', 'mansa_tea', 'thesocialteahouse', 'coffee_bean_leaf_tea', 'feelgoodteaco',
               'senzatea', 'waterfallteas', 'zealongtea', 'thehealingteaco_', 'teaculturesofficial', 'elephantorigins',
               'mamancyteaco', 'mennotea', 'teabotanics', '1852ceylonvalleytea', 'goldenmoontea', 'tealife.com.au',
               'yourteaoflife', 'ghograjanteas', 'allanscoffee', 'dwellteaco', 'fraser.tea', 'jinx.tea',
               'teaswithmeaning', 'tarltontea.lk', 'dilstea', 'chadotea', 'chayamtea', 'gopaldharateas',
               'higherlivingtea', 'gyngertea', 'ivysteaco', 'pampatea', 'nagomitea.cz', 'orijintea', 'matchatea.cz',
               'vypij.cz', 'rising.tea', 'mixteecz', 'whitepeonytea', 'pomoc_z_prirody', 'cgfoodscz',
               'pauwex_walachian_tea', 'cajove_bedynky', 'cajonara', 'biocaj_english_tea_shop.cz', 'oxalis_beroun',
               'herbea.cz', 'pangeatea.cz']
    for seller in sellers:
        posts = instaloader.Profile.from_username(L.context, seller).get_posts()
        path_seller = Path('raw_data', seller)
        filenames = next(walk(path_seller), (None, None, []))[2]
        SINCE = datetime(2015, 5, 1)
        UNTIL = datetime(2015, 3, 1)
        # takewhile(lambda p: p.date > UNTIL, dropwhile(lambda p: p.date > SINCE, posts)):
        for post in posts:
            if post.typename == 'GraphImage':
                download_file = post.date_utc.strftime("%Y-%m-%d_%H-%M-%S_UTC.jpg")
                if download_file in filenames:
                    print('Already Downloaded!')
                else:
                    L.download_post(post, path_seller)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    init()
