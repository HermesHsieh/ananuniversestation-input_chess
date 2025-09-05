上傳index.html流程

以Github的最新版本[index.html]

1. 單獨複製 index.html (Production LIFF_ID) 到 CloudfaireUploadProduction 資料夾
2. 單獨複製 index.html (改Develop LIFF_ID) 到 CloudfaireUploadDevelop 資料夾


3. 先上傳Preview版本到 Cloudfaire

開啟 Cloudfaire 的 Dashboard
> Compute(Workers) > 點即專案名稱 [ 5chess-game ]
> [ Create deployment ]
> 選取 [ Preview ]
> Preview 輸入任意 {PreviewName}


4. 開啟 LINE Developers, 設定LIFF Link

https://developers.line.biz/console/provider/2003267109

> [ 5 Chess Game Dev ] > [ LIFF ] > [ LIFF detail ]

設定機器人 5ChessNameDev LIFF 的 Endpoint URL

Cloudfaire Preview的路徑 https://develop.5chess-game.pages.dev

原始預設 https://hermeshsieh.github.io/ananuniversestation-input_chess/index.html

5. 開啟 LINE機器人 [HealRecorder Dev]

6. 傳送訊息 [ https://liff.line.me/2007931033-4D50LqwM ]
用以開啟測試的LIFF連結

7. 驗證頁面是否正確, 獲取LineID

正確無誤後

8. [ Create deployment ] > 選取 [ Production ]

上傳成功 > 回到正式版

9. 開啟正式版 LINE機器人 [HealRecorder]

測試 [ 象卦AI 療癒算分 ] 

正確無誤即可