import numpy as np
from numpy.testing import assert_array_equal
import six
import pytest
from text.KeywordExtractor import KeywordExtractor

data = [
    '缺圖<箱購>【熊寶貝】衣物柔軟精純淨溫和 3.2L x 4入組',
    '<超值7件組>【白蘭】超濃縮洗衣精1瓶+6 補充包(2.7Kg x1瓶+1.6Kg x6包)(蘆薈清淨)',
    '熊寶貝 柔軟護衣精(2018新包裝)(沁藍海洋香3.2L)',
    '【白蘭】含熊寶貝馨香精華大自然馨香洗衣粉 4.25kg(2入)',
    '【LUX 麗仕】柔亮絲滑潤髮乳 NEW 750ml',
    '【1/2短效期】【DOVE 多芬】清爽水嫩潔膚塊 4入',
    '【白蘭】茶樹除菌超濃縮洗衣精 2. 7Kg',
    '<TreeMall 來店禮獨家組>【白蘭】含熊寶貝超濃縮洗衣精 1+3件組(純淨溫和)',
    '【白蘭】茶樹除菌洗衣粉4.25kg',
    '【白蘭】強效除？過敏洗衣粉 4.25kg',
    '【白蘭】動力配方洗碗精(鮮柚)1kg',
    '【熊寶貝】衣物香氛袋清香21g',
    '<4入瓶裝箱購 贈購物袋>【熊寶貝】衣物柔軟精3.0L/3.2Lx4(七款選)(沁藍海洋香 3.2L)',
    '<超值組>【DOMESTOS 多霸道】多功能除菌清潔劑500ml x 2',
    '<箱購> 【立頓】黃牌精選紅茶 200G x 36入組',
    '【LUX 麗仕】絲蛋白精華沐浴乳水嫩柔膚1L',
    '<TreeMall 來店禮獨家組>【白蘭】含熊寶貝超濃縮洗衣精 1+3件組(花漾清新)',
    '【超值任選】【DOVE 多芬】水潤植萃潤髮乳  玫瑰精華 500ml',
    '<超值7件組>【白蘭】超濃縮洗衣精1瓶+6 補充包(2.7Kg x1瓶+1.6Kg x6包)(強效潔淨除蹣)',
    '【熊寶貝】衣物香氛袋薰衣21g',
    '<TreeMall 來店禮獨家組>【白蘭】含熊寶貝超濃縮洗衣精 1+3件組(大自然馨香)',
    '<超值12件組>【白蘭】不含熊濃縮洗衣精12送6組(1.6kg x12包)送衛生紙6包(蘆薈親膚)',
    '【蒂沐蝶 歐洲天然有機洗沐獨家組】深層純淨1+4超值組+新款植萃皂 贈荷木方形梳(玫瑰保濕皂)',
    '<超值7件組>【白蘭】超濃縮洗衣精1瓶+6 補充包(2.7Kg x1瓶+1.6Kg x6包)(茶樹除菌)',
    '<箱購>【白蘭】強效潔淨除\uee80超濃縮洗衣粉1.9kg x 9入',
    '<超值組>【白蘭】超濃縮洗衣精1+6件組(2.7Kg x1瓶+1.6Kg x6包)',
    '【DOVE 多芬】滋養柔膚沐浴乳 舒敏溫和配方 補充包 2017版  650g',
    '<1+2組>【白蘭】超濃縮洗衣精1+2補(2.7kg x1+1.6kg x2包)(強效潔淨除蹣)',
    '【白蘭】蘆薈親膚超濃縮洗衣精 2. 7Kg',
    '<超值組>【白蘭】蘆薈親膚超濃縮洗衣精1+6件組(2.7Kg x1瓶+1.6Kg x6包)',
    '【康寶】濃湯-自然原味銀魚海帶芽 2*37g',
    '箱購【白蘭】陽光馨香超濃縮洗衣精補充包 1.6Kgx8入',
    '獨家贈【DOVE 多芬】滋養柔膚沐浴乳 滋養柔嫩配方(1Lx1+650mlx5)',
    '【Timotei 蒂沐蝶 】深層純淨護髮乳 500g',
    '【白蘭】含熊寶貝馨香呵護精華純淨溫和洗衣精補充包 1.65kg',
    '東森獨家【白蘭】陽光馨香洗衣粉超值組(東森獨規)',
    '【諾淨】酵素低敏濃縮洗衣精 (護色) 1.5L',
    '<超值7件組>【白蘭】含熊寶貝超濃縮洗衣精 1+6件組 (大自然馨香)',
    '【LUX 麗仕】煥膚香皂煥活冰爽 6入 85g',
    '<箱購>【白蘭】陽光馨香超濃縮洗衣精 2.7Kg  x 4入組',
    '<1+2組>【白蘭】超濃縮洗衣精1+2補(2.7kg x1+1.6kg x2包)(陽光馨香)',
    '【Simple】清妍清新舒緩潔面露 50ML+ 清妍旅行組(2x50ml)',
    '【白蘭】動力配方洗碗精(檸檬)2.8kg',
    '<超值12件組>【白蘭】不含熊濃縮洗衣精12送6組(1.6kg x12包)送衛生紙6包(錯誤)',
    '<箱購>【白蘭】強效除\uee80過敏洗衣粉 4.25kg x 4入組',
    '【潔而亮】特強去污液(清新檸檬芬芳)500ml',
    '【LUX 麗仕】精油香氛沐浴乳迷醉甜香1L',
    '<超值7件組>【白蘭】含熊寶貝超濃縮洗衣精 1+6件組 (2.8Kg x1瓶+1.65 x6包)(森林晨露)',
    '【白蘭】含熊寶貝馨香精華大自然馨香超濃縮洗衣精 2.8kg*1 +補充包1.65kg*2',
    '熊寶貝 柔軟護衣精(2018新包裝)(淡雅櫻花香3.0L)',
    '<箱購>【AXE】黯黑經典香體噴霧150ml x 6入',
    '【白蘭】含熊寶貝馨香精華純淨溫和超濃縮洗衣精 1+9件組(2.8Kg x1瓶+1.65Kg x9包)',
    '【DOVE 多芬】舒柔水嫩沐浴乳(新)  1000ML',
    '<箱購> 【立頓】奶茶粉原味罐裝 450g x 12入組',
    '【LUX 麗仕】日本極致修護髮膜 200g'
]


@pytest.mark.parametrize('n', range(1,11,2))
def test_transform_basic(n):
    ke = KeywordExtractor(n_keyword=n)
    ke.fit(data)
    result = ke.transform(data)
    shape = result.shape
    assert shape[0] == len(data)
    assert shape[1] == n
    assert np.issubdtype(result.dtype, np.dtype('U'))

    
@pytest.mark.parametrize('n', range(1,11,2))
def test_transform_proba_shape(n):
    ke = KeywordExtractor(n_keyword=n)
    ke.fit(data)
    result = ke.transform_proba(data)
    shape = result.shape
    assert shape[0] == len(data)
    assert shape[1] == n
    assert np.issubdtype(result.dtype, np.dtype('int'))
    
    
def test_enable_english():
    ke_enable = KeywordExtractor(enable_english=True)
    ke_enable.fit(data)
    result = ke_enable.transform('【CLEAR 淨】 無MIT 多效水護洗髮乳 750ML')
    assert_array_equal(
        result,
        np.array(['髮乳', 'MIT', 'CLEAR', '', ''])
    )
    
    ke_disable = KeywordExtractor(enable_english=False)
    ke_disable.fit(data)
    result = ke_disable.transform('【CLEAR 淨】 無MIT 多效水護洗髮乳 750ML')
    assert_array_equal(
        result,
        np.array(['髮乳', '', '', '', ''])
    )



def test_min_ch_keyword_len():
    ke = KeywordExtractor(min_ch_keyword_len=3)
    ke.fit(data)
    test_string = '【白蘭】含熊寶貝馨香精華花漾清新洗衣粉4.25kg'
    result = ke.transform(test_string)
    assert_array_equal(
        result,
        np.array(['洗衣粉', '含熊寶貝', '含熊寶貝馨香精華', 
                  '熊寶貝', '花漾清新'])
    )

def test_string_input():
    ke = KeywordExtractor(n_keyword=5)
    ke.fit(data)
    test_string = '【白蘭】含熊寶貝馨香精華花漾清新洗衣粉4.25kg' 
    result = ke.transform(test_string)
    assert_array_equal(
        result,
        np.array(['白蘭', '洗衣', '洗衣粉', '馨香', '含熊寶貝'])
    )


def test_no_duplicated_keywords():
    ke = KeywordExtractor(n_keyword=5)
    ke.fit(data)
    test_string = '洗衣洗衣洗衣洗衣'
    result = ke.transform(test_string)
    assert_array_equal(
        result,
        np.array(['洗衣', '', '', '', ''])
    )


def test_empty_input():
    ke = KeywordExtractor(n_keyword=5)
    ke.fit(data)
    test_string = ''
    result = ke.transform(test_string)
    assert_array_equal(
        result,
        np.array(['', '', '', '', ''])
    )