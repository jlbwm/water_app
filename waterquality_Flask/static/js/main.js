/* ========================================================================= */
/*	Preloader
/* ========================================================================= */
jQuery(window).load(function () {

    $("#preloader").fadeOut("slow");
});

/* ========================================================================= */
/*  Welcome Section Slider
/* ========================================================================= */
// $(document).ready(function() { ... })
$(function () {
    var Page = (function () {
        var $navArrows = $('#nav-arrows'),
            $nav = $('#nav-dots > span'),
            slitslider = $('#slider').slitslider({
                onBeforeChange: function (slide, pos) {
                    $nav.removeClass('nav-dot-current');
                    $nav.eq(pos).addClass('nav-dot-current');
                }
            }),
            init = function () {
                initEvents();
            },
            initEvents = function () {
                // add navigation events
                $navArrows.children(':last').on('click', function () {
                    slitslider.next();
                    return false;

                });
                $navArrows.children(':first').on('click', function () {
                    slitslider.previous();
                    return false;

                });
                $nav.each(function (i) {
                    $(this).on('click', function (event) {
                        var $dot = $(this);
                        if (!slitslider.isActive()) {
                            $nav.removeClass('nav-dot-current');
                            $dot.addClass('nav-dot-current');
                        }
                        slitslider.jump(i + 1);
                        return false;
                    });
                });
            };

        return { init: init };
    })();
    Page.init();
});

const formUpload = function() {
    // UPLOAD CLASS DEFINITION

    var dropZone = document.getElementById('drop-zone');
    var uploadForm = document.getElementById('js-upload-form');

    var startUpload = function (files) {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            let file = files[i];
          
            // Check the file type.
            if (!file.type.match('image.*')) {
              continue;
            }
            // Add the file to the request.
            formData.append('images', file);
          }
        $.ajax({
            xhr: function() {
                const xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener("progress", e => {
                    if (e.lengthComputable) {
                        // const percentComplete = evt.loaded / evt.total;
                        console.log(e);
                    }
               }, false);
        
               xhr.addEventListener("progress", e => {
                   if (e.lengthComputable) {
                    //    const percentComplete = evt.loaded / evt.total;
                       //Do something with download progress
                       console.log(e);
                   }
               }, false);
               return xhr;
            },
            url: "/webAPI",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,

            success: data => {
                if (data.error) {
                    console.log("handle error");
                } else {
                    // carousel clear
                    $('.carousel-inner').empty();
                    $('.carousel-indicators').empty();
                    data.results.forEach((img, index) => {
                        // set active for the first item
                        if (index == 0) {
                            $('<div class="item active"><img class="img-responsive" src="' + img.thumbnail + '"><div class="carousel-caption"><h3>' + img.prediction + '</h3><p>' + img.time  + ' ' + img.location + '</p></div></div>').appendTo('.carousel-inner')
                        } else {
                            $('<div class="item"><img class="img-responsive" src="' + img.thumbnail + '"><div class="carousel-caption"><h3>' + img.prediction + '</h3><p>' + img.time  + ' ' + img.location + '</p></div></div>').appendTo('.carousel-inner')
                        
                        }
                        $('<li data-target="#carousel-generic" data-slide-to="' + index + '"></li>').appendTo('.carousel-indicators')
                    });
                }
            }
        })
    }

    uploadForm.addEventListener('submit', function (e) {
        var uploadFiles = document.getElementById('js-upload-files').files;
        e.preventDefault()

        startUpload(uploadFiles)
    })

    dropZone.ondrop = function (e) {
        e.preventDefault();
        this.className = 'upload-drop-zone';

        startUpload(e.dataTransfer.files)
    }
    dropZone.ondragover = function () {
        this.className = 'upload-drop-zone drop';
        return false;
    }
    dropZone.ondragleave = function () {
        this.className = 'upload-drop-zone';
        return false;
    }

};
formUpload();


$(document).ready(function() {

    /* ========================================================================= */
	/*	Menu item highlighting
	/* ========================================================================= */

    jQuery('#nav').singlePageNav({
        offset: jQuery('#nav').outerHeight(),
        filter: ':not(.external)',
        speed: 2000,
        currentClass: 'current',
        easing: 'easeInOutExpo',
        updateHash: true,
        beforeStart: function () {
            console.log('begin scrolling');
        },
        onComplete: function () {
            console.log('done scrolling');
        }
    });

    $(window).scroll(function () {
        if ($(window).scrollTop() > 400) {
            $(".navbar-brand a").css("color", "#fff");
            $("#navigation").removeClass("animated-header");
        } else {
            $(".navbar-brand a").css("color", "inherit");
            $("#navigation").addClass("animated-header");
        }
    });

    /* ========================================================================= */
	/*	Fix Slider Height
	/* ========================================================================= */

    // Slider Height
    var slideHeight = $(window).height();

    $('#home-slider, #slider, .sl-slider, .sl-content-wrapper').css('height', slideHeight);

    $(window).resize(function () {
        'use strict',
            $('#home-slider, #slider, .sl-slider, .sl-content-wrapper').css('height', slideHeight);
    });

    $("#works, #testimonial").owlCarousel({
        navigation: true,
        pagination: false,
        slideSpeed: 700,
        paginationSpeed: 400,
        singleItem: true,
        navigationText: ["<i class='fa fa-angle-left fa-lg'></i>", "<i class='fa fa-angle-right fa-lg'></i>"]
    });

    /* ========================================================================= */
	/*	Featured Project Lightbox
	/* ========================================================================= */

    $(".fancybox").fancybox({
        padding: 0,

        openEffect: 'elastic',
        openSpeed: 650,

        closeEffect: 'elastic',
        closeSpeed: 550,

        closeClick: true,

        beforeShow: function () {
            this.title = $(this.element).attr('title');
            this.title = '<h3>' + this.title + '</h3>' + '<p>' + $(this.element).parents('.portfolio-item').find('img').attr('alt') + '</p>';
        },

        helpers: {
            title: {
                type: 'inside'
            },
            overlay: {
                css: {
                    'background': 'rgba(0,0,0,0.8)'
                }
            }
        }
    });

    $('.collapse').collapse();
});

var wow = new WOW({
    offset: 75,          // distance to the element when triggering the animation (default is 0)
    mobile: false,       // trigger animations on mobile devices (default is true)
});
wow.init();

