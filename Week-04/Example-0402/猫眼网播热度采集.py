from selenium import webdriver

if __name__ == "__main__":
    browser = webdriver.Chrome(executable_path=r"D:\Python38_64\chromedriver_81_0_4044_69.exe")  # ChromeDriver可执行文件的路径
    browser.get("http://piaofang.maoyan.com/dashboard/web-heat")
    for movie_label in browser.find_elements_by_css_selector(
            "#app > div > div > div.dashboard-content > div.dashboard-list.dashboard-left.bg > div.movielist-container > div > table > tbody > tr"):
        print("排名:", movie_label.find_element_by_class_name("moviename-index").text)
        print("名称:", movie_label.find_element_by_class_name("moviename-name").text)
        print("信息:", movie_label.find_element_by_class_name("moviename-info").text)
        print("信息:", movie_label.find_element_by_class_name("heat-text").text)
        print("信息:", movie_label.find_element_by_class_name("last-col").text)
