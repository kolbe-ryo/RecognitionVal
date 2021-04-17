# RecognitionVal
本コードによってデジタルメータの数値認識が可能となる。

## 1. 元データの取得
<img width="455" alt="Screen Shot 2021-04-18 at 2 44 56" src="https://user-images.githubusercontent.com/77920313/115121915-1c423300-9ff0-11eb-912c-87a29019b6de.png">

## 2. ブラー、グレースケール化処理
<img width="455" alt="Screen Shot 2021-04-18 at 2 46 59" src="https://user-images.githubusercontent.com/77920313/115121951-5dd2de00-9ff0-11eb-8713-dc45183789b8.png">

## 3. 2値化処理、膨張収縮、Erodeなどノイズ除去処理
<img width="455" alt="Screen Shot 2021-04-18 at 2 48 06" src="https://user-images.githubusercontent.com/77920313/115121982-82c75100-9ff0-11eb-8eb6-02fb89014b17.png">

## 4. Canny処理による輪郭取得、サイズ／アスペクト比／連続性による抽出
<img width="455" alt="Screen Shot 2021-04-18 at 2 49 56" src="https://user-images.githubusercontent.com/77920313/115122032-c457fc00-9ff0-11eb-9dae-14a9bdf1aed5.png">

## 5. 数値のみ抽出
<img width="533" alt="Screen Shot 2021-04-18 at 2 53 00" src="https://user-images.githubusercontent.com/77920313/115122116-32042800-9ff1-11eb-8b07-b1b0c553471c.png">

## 6. セグメント認識
<img width="533" alt="Screen Shot 2021-04-18 at 2 53 54" src="https://user-images.githubusercontent.com/77920313/115122141-5233e700-9ff1-11eb-8f27-1f412f63dea9.png">
