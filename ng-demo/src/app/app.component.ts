import { Component, Sanitizer } from '@angular/core';
import { Camera, CameraResultType } from '@capacitor/camera';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer } from '@angular/platform-browser';

type PythonRes = {
  caption: String
}

type Image = {
  image_url: any
  caption: String
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent {
  title = 'caption-images';
  api_addr = 'http://127.0.0.1:5000/caption'
  images: Image[] = [];
  
  constructor (private http: HttpClient, private sanitizer: DomSanitizer) { }

  generateCaption(base64Image: String) {
    return new Promise<String>((resolve, reject) => {
      this.http.post(this.api_addr, base64Image).subscribe((res) => {
        resolve((res as PythonRes).caption)
      })
    })
  }

  removeImg(image_url: String) {
    this.images = this.images.filter((image) => image.image_url !== image_url)
  }

  async takePicture() {
    let image: Image = {'image_url':'', 'caption': ''}

    let raw_image = await Camera.getPhoto({
      quality: 90,
      allowEditing: true,
      resultType: CameraResultType.Base64
    });

    image.image_url = this.sanitizer.bypassSecurityTrustResourceUrl(URL.createObjectURL(await fetch(`data:image/jpeg;base64,${raw_image.base64String}`).then((res) => res.blob())));
    
    (document.getElementById('LoaderLauncher') as HTMLButtonElement).click();

    image.caption = await this.generateCaption((raw_image.base64String as String));

    (document.getElementById('LoaderLauncher') as HTMLButtonElement).click();
    this.images.push(image)
  }
  
}